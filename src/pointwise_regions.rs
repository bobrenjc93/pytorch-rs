//! Logical numerical regions within one native launch.
//!
//! Realization and fusion are separate: an intermediate becomes a rounding
//! boundary only when its producer and consumer finish in different regions.
//! The scalar planner follows the default CUDA pointwise realization/scheduling
//! rules (CSE operation/read counts, locality order, and stable fusion). It has
//! no graph-size certificate or joint-lowering fallback.
use super::{Graph, Node, indexing::Address};
use std::collections::{BTreeSet, HashMap, HashSet};

/// One graph-rewrite interpretation shared by planning and numerical lowering.
/// Independent tensor producers stay distinct; only proven graph aliases map
/// to an earlier SSA value. Scalar CSE belongs inside each numerical region.
#[derive(Clone, Debug)]
pub(crate) struct Canonical {
    pub mapped: Vec<usize>,
    pub nodes: Vec<Node>,
}

impl Canonical {
    /// Use the same canonical dependencies and liveness as numerical planning.
    pub(super) fn uses_erf(&self, outputs: &[usize]) -> bool {
        let roots: Vec<_> = outputs.iter().map(|&id| self.mapped[id]).collect();
        locality_order(&self.nodes, &roots)
            .0
            .into_iter()
            .any(|id| matches!(self.nodes[id], Node::Erf(_)))
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Plan {
    pub canonical: Canonical,
    pub regions: Vec<Region>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Region {
    /// Original SSA values stored publicly or consumed by a later region.
    pub roots: Vec<usize>,
    /// Original SSA values imported as already-rounded per-lane values.
    pub imports: Vec<usize>,
}

fn operands(node: &Node) -> Vec<usize> {
    match *node {
        Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) => vec![a, b],
        Node::Neg(a) | Node::Relu(a) | Node::Sin(a) | Node::Cos(a) | Node::Erf(a) => vec![a],
        _ => vec![],
    }
}

fn tensor_operation(node: &Node) -> bool {
    matches!(
        node,
        Node::Add(..)
            | Node::Sub(..)
            | Node::Mul(..)
            | Node::Neg(_)
            | Node::Relu(_)
            | Node::Sin(_)
            | Node::Cos(_)
            | Node::Erf(_)
    )
}

fn append_unique(target: &mut Vec<usize>, values: impl IntoIterator<Item = usize>) {
    for value in values {
        if !target.contains(&value) {
            target.push(value);
        }
    }
}

/// Preserve default-inference tensor dependencies through realization. Two
/// equal operations are not the same tensor producer: merging them here can
/// introduce an import of a rounded value where the reference recomputes it.
/// Expression interning for operation counts and region-local CSE are separate.
pub(crate) fn canonical_nodes(graph: &Graph) -> Canonical {
    let (aliases, zeros) = super::lowering::early_aliases(graph);
    let mut mapped = Vec::with_capacity(graph.nodes.len());
    let mut nodes = graph.nodes.clone();
    for (id, node) in graph.nodes.iter().enumerate() {
        if aliases[id] != id {
            mapped.push(mapped[aliases[id]]);
            continue;
        }
        let key = if zeros[id] {
            // Tensor zero producers retain a tensor node, independently of
            // scalar literals. Their scalar expression is counted below.
            Node::Mul(usize::MAX, usize::MAX)
        } else {
            super::lowering::remap(node, &mapped)
        };
        mapped.push(id);
        nodes[id] = key;
    }
    Canonical { mapped, nodes }
}

fn node_operands(node: &Node) -> Vec<usize> {
    if matches!(node, Node::Mul(usize::MAX, usize::MAX)) {
        vec![]
    } else {
        operands(node)
    }
}

/// Simulate the mutable reverse traversal in `post_grad.reorder_for_locality`.
/// A producer moves immediately before a consumer only once every user has
/// been seen. The output sentinel's operands retain observable return order.
fn locality_order(nodes: &[Node], outputs: &[usize]) -> (Vec<usize>, Vec<Vec<usize>>) {
    let count = nodes.len();
    let mut live = vec![false; count];
    let mut pending = outputs.to_vec();
    while let Some(id) = pending.pop() {
        if !live[id] {
            live[id] = true;
            pending.extend(node_operands(&nodes[id]));
        }
    }
    let mut users = vec![Vec::new(); count];
    for (id, node) in nodes.iter().enumerate() {
        if live[id] {
            for a in node_operands(node) {
                append_unique(&mut users[a], [id]);
            }
        }
    }
    for &id in outputs {
        append_unique(&mut users[id], [count]);
    }
    let mut order: Vec<_> = (0..count).filter(|&id| live[id]).collect();
    order.push(count);
    let mut seen = vec![false; count + 1];
    let mut current = Some(count);
    while let Some(id) = current {
        seen[id] = true;
        let args = if id == count {
            outputs.to_vec()
        } else {
            node_operands(&nodes[id])
        };
        for a in args {
            if tensor_operation(&nodes[a]) && users[a].iter().all(|&u| seen[u]) {
                let position = order.iter().position(|&x| x == a).unwrap();
                order.remove(position);
                let destination = order.iter().position(|&x| x == id).unwrap();
                order.insert(destination, a);
            }
        }
        let position = order.iter().position(|&x| x == id).unwrap();
        current = position.checked_sub(1).map(|p| order[p]);
    }
    order.pop();
    (order, users)
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Shape {
    full: bool,
    axes: BTreeSet<(usize, usize)>,
}

impl Shape {
    fn scalar() -> Self {
        Self {
            full: false,
            axes: BTreeSet::new(),
        }
    }
    fn input(address: &Address) -> Self {
        match address {
            Address::Linear => Self {
                full: true,
                axes: BTreeSet::new(),
            },
            Address::Broadcast(terms) => Self {
                full: false,
                axes: terms.iter().map(|&(d, n, _)| (d, n)).collect(),
            },
        }
    }
    fn merge(&mut self, other: &Self) {
        self.full |= other.full;
        self.axes.extend(&other.axes);
        if self.full {
            self.axes.clear();
        }
    }
    fn elements(&self, hint: u64) -> u64 {
        if self.full {
            hint
        } else {
            self.axes
                .iter()
                .fold(1u64, |n, &(_, d)| n.saturating_mul(d as u64))
        }
    }
}

#[derive(Clone, Default)]
struct Expression {
    value: usize,
    operations: BTreeSet<usize>,
    // In scalar evaluation order, including distinct realized buffers.
    reads: Vec<usize>,
}

fn intern(key: Node, table: &mut HashMap<Node, usize>) -> usize {
    let next = table.len();
    *table.entry(key).or_insert(next)
}

fn load(buffer: usize, table: &mut HashMap<Node, usize>) -> Expression {
    let value = intern(Node::Input(buffer), table);
    Expression {
        value,
        operations: BTreeSet::from([value]),
        reads: vec![buffer],
    }
}

#[derive(Clone)]
struct Unit {
    root: usize,
    reads: Vec<usize>,
    shape: Shape,
    public: bool,
}

fn realize(
    id: usize,
    graph: &Graph,
    expressions: &mut [Expression],
    shapes: &[Shape],
    realized: &mut [Option<usize>],
    units: &mut Vec<Unit>,
    table: &mut HashMap<Node, usize>,
) {
    if realized[id].is_some() {
        return;
    }
    let unit = units.len();
    units.push(Unit {
        root: id,
        reads: expressions[id].reads.clone(),
        shape: shapes[id].clone(),
        public: false,
    });
    realized[id] = Some(unit);
    expressions[id] = load(graph.inputs + unit, table);
}

fn realization(
    graph: &Graph,
    addresses: &[Address],
    hint: u64,
    outputs: &[usize],
    nodes: &[Node],
    scalar_output: bool,
) -> (Vec<Unit>, Vec<Option<usize>>) {
    let (order, users) = locality_order(nodes, outputs);
    let mut expressions = vec![Expression::default(); nodes.len()];
    // Tensor extents follow the original operations even when an algebraic
    // identity removes every read (for example x * integer-zero).
    let mut shapes: Vec<Shape> = Vec::with_capacity(nodes.len());
    for node in &graph.nodes {
        let mut shape = Shape::scalar();
        if let Node::Input(input) = *node {
            shape = Shape::input(&addresses[input]);
        } else {
            for a in operands(node) {
                shape.merge(&shapes[a]);
            }
        }
        shapes.push(shape);
    }
    let mut realized = vec![None; nodes.len()];
    let mut units = Vec::new();
    let mut table = HashMap::new();
    let output_set: HashSet<_> = outputs.iter().copied().collect();
    let mut buffer_shapes: Vec<_> = addresses.iter().map(Shape::input).collect();
    for id in order {
        let node = &nodes[id];
        let args = node_operands(node);
        let mut expr = Expression::default();
        for &a in &args {
            expr.operations.extend(&expressions[a].operations);
            append_unique(&mut expr.reads, expressions[a].reads.iter().copied());
        }
        if let Node::Input(input) = *node {
            shapes[id] = Shape::input(&addresses[input]);
            expr = load(input, &mut table);
        } else {
            let key = if matches!(node, Node::Mul(usize::MAX, usize::MAX)) {
                Node::Constant(0)
            } else {
                let values: Vec<_> = expressions.iter().map(|e| e.value).collect();
                super::lowering::remap(node, &values)
            };
            expr.value = intern(key, &mut table);
            expr.operations.insert(expr.value);
        }
        expressions[id] = expr;
        if !tensor_operation(node) {
            continue;
        }
        let expr = &expressions[id];
        let count = expr.operations.len();
        let read_count = expr.reads.len();
        let nontrivial = expr
            .reads
            .iter()
            .filter(|&&buffer| buffer_shapes[buffer].elements(hint) > 1)
            .count();
        let returned = output_set.contains(&id);
        let shared = users[id].len() > 1;
        let cheap = read_count == 0 && count <= 30;
        // Match GraphLowering / StorageBox order. Reuse and the >100-op
        // recursion safeguard are unconditional; accumulated-read realization
        // is a hint and does not realize constant-index scalar expressions.
        if (returned && !cheap && (!scalar_output || shared))
            || (shared && (count > 30 || read_count > 4))
            || ((count > 30 || read_count > 8) && nontrivial > 1)
            || count > 100
        {
            realize(
                id,
                graph,
                &mut expressions,
                &shapes,
                &mut realized,
                &mut units,
                &mut table,
            );
            buffer_shapes.push(shapes[id].clone());
        }
    }
    for &id in outputs {
        if realized[id].is_none() {
            realize(
                id,
                graph,
                &mut expressions,
                &shapes,
                &mut realized,
                &mut units,
                &mut table,
            );
            buffer_shapes.push(shapes[id].clone());
        }
    }
    for &id in outputs {
        units[realized[id].unwrap()].public = true;
    }
    (units, realized)
}

#[derive(Clone)]
struct Group {
    units: Vec<usize>,
    reads: Vec<usize>,
    used: Vec<usize>,
    min: usize,
    max: usize,
}

impl Group {
    fn new(members: Vec<usize>, units: &[Unit], inputs: usize) -> Self {
        // FusedSchedulerNode preserves left/right concatenation. Sorting here
        // changes scalar materialization order even though dependency sets and
        // the group's min/max priorities would remain identical.
        let mut reads = Vec::new();
        let mut used = Vec::new();
        for &id in &members {
            append_unique(&mut reads, units[id].reads.iter().copied());
            append_unique(&mut used, units[id].reads.iter().copied());
            append_unique(&mut used, [inputs + id]);
        }
        reads.retain(|&b| b < inputs || !members.contains(&(b - inputs)));
        Self {
            min: *members.iter().min().unwrap(),
            max: *members.iter().max().unwrap(),
            units: members,
            reads,
            used,
        }
    }
}

fn shared_bytes(a: &Group, b: &Group, sizes: &[u64]) -> u64 {
    a.used
        .iter()
        .filter(|v| b.used.contains(v))
        .fold(0u64, |n, &v| n.saturating_add(sizes[v].saturating_mul(4)))
}

fn distance(a: &Group, b: &Group) -> usize {
    a.min.abs_diff(b.max).max(b.min.abs_diff(a.max))
}

fn cyclic_merge(
    a: usize,
    b: usize,
    groups: &[Option<Group>],
    owner: &[usize],
    inputs: usize,
) -> bool {
    let mut pending = Vec::new();
    for group in [a, b] {
        for &read in &groups[group].as_ref().unwrap().reads {
            if read >= inputs {
                let parent = owner[read - inputs];
                if parent != a && parent != b {
                    pending.push(parent);
                }
            }
        }
    }
    let mut seen = HashSet::new();
    while let Some(group) = pending.pop() {
        if group == a || group == b {
            return true;
        }
        if seen.insert(group) {
            for &read in &groups[group].as_ref().unwrap().reads {
                if read >= inputs {
                    pending.push(owner[read - inputs]);
                }
            }
        }
    }
    false
}

fn can_fuse(
    a: usize,
    b: usize,
    groups: &[Option<Group>],
    owner: &[usize],
    sizes: &[u64],
    inputs: usize,
    single_user: &[bool],
) -> bool {
    if a == b {
        return false;
    }
    let left = groups[a].as_ref().unwrap();
    let right = groups[b].as_ref().unwrap();
    if left.units.len() + right.units.len() > 64 {
        return false;
    }
    let bytes = shared_bytes(left, right, sizes);
    if bytes == 0 {
        return false;
    }
    // Mirror the default peak-memory heuristic over reusable internal inputs.
    // Native pointwise storage has one device/dtype/alignment/stream, so its
    // remaining reuse-key component is the contiguous storage element count.
    let reuse_sizes = |g: &Group| -> BTreeSet<u64> {
        g.reads
            .iter()
            .filter(|&&r| r >= inputs && single_user[r - inputs])
            .map(|&r| sizes[r])
            .collect()
    };
    let lhs = reuse_sizes(left);
    let rhs = reuse_sizes(right);
    let overhead = lhs
        .intersection(&rhs)
        .fold(0u64, |n, &size| n.saturating_add(size));
    if overhead > bytes.saturating_mul(32) {
        return false;
    }
    let vertical = right
        .reads
        .iter()
        .any(|&r| r >= inputs && owner[r - inputs] == a)
        || left
            .reads
            .iter()
            .any(|&r| r >= inputs && owner[r - inputs] == b);
    if !vertical && (bytes < 10 || distance(left, right) > 64) {
        return false;
    }
    !cyclic_merge(a, b, groups, owner, inputs)
}

fn schedule(units: &[Unit], addresses: &[Address], hint: u64) -> Vec<Group> {
    let inputs = addresses.len();
    let sizes: Vec<_> = addresses
        .iter()
        .map(|a| Shape::input(a).elements(hint))
        .chain(units.iter().map(|u| u.shape.elements(hint)))
        .collect();
    let mut user_counts: Vec<usize> = units.iter().map(|u| usize::from(u.public)).collect();
    for unit in units {
        for &read in &unit.reads {
            if read >= inputs {
                user_counts[read - inputs] += 1;
            }
        }
    }
    let single_user: Vec<_> = user_counts.iter().map(|&n| n == 1).collect();
    let mut groups: Vec<_> = (0..units.len())
        .map(|id| Some(Group::new(vec![id], units, inputs)))
        .collect();
    let mut owner: Vec<_> = (0..units.len()).collect();
    for _ in 0..10 {
        let mut ordered: Vec<_> = (0..groups.len())
            .filter(|&id| groups[id].is_some())
            .collect();
        ordered.sort_by_key(|&id| groups[id].as_ref().unwrap().min);
        let mut buffers: Vec<(usize, Vec<usize>)> = Vec::new();
        for &id in &ordered {
            for &buffer in &groups[id].as_ref().unwrap().used {
                if let Some((_, users)) = buffers.iter_mut().find(|(b, _)| *b == buffer) {
                    users.push(id);
                } else {
                    buffers.push((buffer, vec![id]));
                }
            }
        }
        let mut seen = HashSet::new();
        let mut candidates = Vec::new();
        for (_, users) in buffers {
            for (position, &a) in users.iter().enumerate() {
                for &b in users.iter().skip(position + 1).take(64) {
                    if seen.insert((a, b))
                        && can_fuse(a, b, &groups, &owner, &sizes, inputs, &single_user)
                    {
                        let left = groups[a].as_ref().unwrap();
                        let right = groups[b].as_ref().unwrap();
                        candidates.push((
                            left.units[0],
                            right.units[0],
                            shared_bytes(left, right, &sizes),
                            distance(left, right),
                        ));
                    }
                }
            }
        }
        // Stable sort preserves the reference's buffer-group/pair traversal
        // when shared-byte and proximity scores tie.
        candidates.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.3.cmp(&b.3)));
        let mut merged = false;
        for (u, v, _, _) in candidates {
            let (a, b) = (owner[u], owner[v]);
            if can_fuse(a, b, &groups, &owner, &sizes, inputs, &single_user) {
                let mut members = groups[a].take().unwrap().units;
                members.extend(groups[b].take().unwrap().units);
                for &id in &members {
                    owner[id] = a;
                }
                groups[a] = Some(Group::new(members, units, inputs));
                merged = true;
            }
        }
        if !merged {
            break;
        }
    }
    // Stable topological emission matters when fusion spans an unrelated unit.
    let mut remaining: Vec<_> = groups.into_iter().flatten().collect();
    remaining.sort_by_key(|g| g.min);
    let mut emitted = vec![false; units.len()];
    let mut result = Vec::new();
    while !remaining.is_empty() {
        let next = remaining
            .iter()
            .position(|g| g.reads.iter().all(|&b| b < inputs || emitted[b - inputs]))
            .expect("acyclic numerical schedule");
        let group = remaining.remove(next);
        for &unit in &group.units {
            emitted[unit] = true;
        }
        result.push(group);
    }
    result
}

/// Plan every admitted bounded graph. `output_order` is a permutation of the
/// canonical output slots; it affects numerical planning, never Graph identity.
/// `numerical_hint` belongs to the logical specialization, while actual sizes
/// remain the indexing/allocation owner's responsibility.
pub(crate) fn plan(
    graph: &Graph,
    addresses: &[Address],
    numerical_hint: u64,
    output_order: &[usize],
    scalar_output: bool,
) -> Plan {
    let canonical = canonical_nodes(graph);
    let Canonical { mapped, nodes } = &canonical;
    let mut outputs = Vec::new();
    for &slot in output_order {
        append_unique(&mut outputs, [mapped[graph.outputs[slot]]]);
    }
    let (units, realized) = realization(
        graph,
        addresses,
        numerical_hint,
        &outputs,
        nodes,
        scalar_output,
    );
    let groups = schedule(&units, addresses, numerical_hint);
    let mut group_of_unit = vec![0; units.len()];
    for (group, info) in groups.iter().enumerate() {
        for &unit in &info.units {
            group_of_unit[unit] = group;
        }
    }
    let mut result = Vec::new();
    for (group_id, group) in groups.iter().enumerate() {
        let mut imports = Vec::new();
        for &read in &group.reads {
            if read >= graph.inputs {
                append_unique(&mut imports, [units[read - graph.inputs].root]);
            }
        }
        let mut roots = Vec::new();
        for &unit in &group.units {
            let needed = units.iter().enumerate().any(|(consumer, info)| {
                group_of_unit[consumer] != group_id && info.reads.contains(&(graph.inputs + unit))
            });
            if needed {
                append_unique(&mut roots, [units[unit].root]);
            }
            for &slot in output_order {
                let original = graph.outputs[slot];
                if realized[mapped[original]] == Some(unit) {
                    append_unique(&mut roots, [original]);
                }
            }
        }
        if !roots.is_empty() {
            result.push(Region { roots, imports });
        }
    }
    Plan {
        canonical,
        regions: result,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(
        graph: &Graph,
        addresses: &[Address],
        hint: u64,
        order: &[usize],
        scalar_output: bool,
    ) -> Vec<Region> {
        super::plan(graph, addresses, hint, order, scalar_output).regions
    }

    fn nonlinear(shared_input: bool) -> Graph {
        Graph {
            inputs: 2,
            nodes: vec![
                Node::Input(0),
                Node::Input(1),
                Node::Mul(0, usize::from(!shared_input)),
                Node::Neg(2),
                Node::Sin(2),
            ],
            outputs: vec![3, 4],
        }
    }

    #[test]
    fn shared_reads_set_horizontal_threshold() {
        let addresses = [Address::Linear, Address::Linear];
        for (one_input, boundary) in [(false, 2), (true, 3)] {
            let graph = nonlinear(one_input);
            assert_eq!(
                plan(&graph, &addresses, 1, &[0, 1], false),
                vec![
                    Region {
                        roots: vec![3],
                        imports: vec![]
                    },
                    Region {
                        roots: vec![4],
                        imports: vec![]
                    }
                ]
            );
            assert_eq!(
                plan(&graph, &addresses, boundary, &[0, 1], false),
                vec![Region {
                    roots: vec![3, 4],
                    imports: vec![]
                }]
            );
        }
    }

    #[test]
    fn seven_sines_do_not_force_joint_lowering() {
        let mut graph = Graph {
            inputs: 2,
            nodes: vec![
                Node::Input(0),
                Node::Input(1),
                Node::Mul(0, 1),
                Node::Sub(2, 0),
            ],
            outputs: vec![3],
        };
        let mut value = 2;
        for _ in 0..7 {
            graph.nodes.push(Node::Sin(value));
            value = graph.nodes.len() - 1;
        }
        graph.outputs.push(value);
        assert_eq!(
            plan(
                &graph,
                &[Address::Linear, Address::Linear],
                1,
                &[0, 1],
                false
            ),
            vec![
                Region {
                    roots: vec![3],
                    imports: vec![]
                },
                Region {
                    roots: vec![value],
                    imports: vec![]
                }
            ]
        );
    }

    fn long_chain() -> Graph {
        let mut graph = Graph {
            inputs: 2,
            nodes: vec![
                Node::Input(0),
                Node::Input(1),
                Node::Mul(0, 1),
                Node::Sub(2, 0),
            ],
            outputs: vec![3],
        };
        let mut value = 2;
        for _ in 0..64 {
            for _ in 0..30 {
                graph.nodes.push(Node::Sin(value));
                value = graph.nodes.len() - 1;
            }
            let base = value;
            graph.nodes.push(Node::Sin(base));
            let sin = graph.nodes.len() - 1;
            graph.nodes.push(Node::Cos(base));
            let cos = graph.nodes.len() - 1;
            graph.nodes.push(Node::Add(sin, cos));
            value = graph.nodes.len() - 1;
        }
        graph.nodes.push(Node::Sin(value));
        graph.outputs.push(graph.nodes.len() - 1);
        graph
    }

    #[test]
    fn return_order_controls_realized_unit_fusion() {
        let graph = long_chain();
        assert_eq!(graph.nodes.len(), 2117);
        let addresses = [Address::Linear, Address::Linear];
        let qr = plan(&graph, &addresses, 2, &[0, 1], false);
        let rq = plan(&graph, &addresses, 2, &[1, 0], false);
        assert!(qr[0].roots.contains(&3));
        assert!(qr[0].roots.len() > 1);
        let q_region = rq.iter().find(|r| r.roots.contains(&3)).unwrap();
        assert_eq!(q_region.roots, vec![3]);
        assert!(q_region.imports.is_empty());
        assert!(!rq[0].roots.contains(&3));
        for regions in [&qr, &rq] {
            let mut available: BTreeSet<usize> = BTreeSet::new();
            for region in regions {
                assert!(region.imports.iter().all(|i| available.contains(i)));
                available.extend(region.roots.iter().copied());
            }
        }
    }

    #[test]
    fn single_read_long_expressions_use_the_hundred_op_safeguard() {
        let mut graph = Graph {
            inputs: 1,
            nodes: vec![Node::Input(0)],
            outputs: vec![],
        };
        for _ in 0..140 {
            graph.nodes.push(Node::Sin(graph.nodes.len() - 1));
        }
        graph.outputs.push(graph.nodes.len() - 1);
        let Canonical { mapped, nodes } = canonical_nodes(&graph);
        let outputs = [mapped[graph.outputs[0]]];
        let (units, _) = realization(&graph, &[Address::Linear], 13, &outputs, &nodes, false);
        // One read plus 100 sine operations first exceeds 100. A single
        // nontrivial read does not turn the >30-op hint into realization.
        assert_eq!(units[0].root, 100);
        assert_eq!(units.len(), 2);
        // Those two units vertically fuse, so there is no rounding boundary.
        assert_eq!(
            plan(&graph, &[Address::Linear], 13, &[0], false),
            vec![Region {
                roots: graph.outputs.clone(),
                imports: vec![]
            }]
        );
    }

    #[test]
    fn zero_read_tensor_outputs_keep_extents_without_cascade_realization() {
        let graph = Graph {
            inputs: 1,
            nodes: vec![
                Node::Input(0),
                Node::Integer(0),
                Node::Mul(0, 1),
                Node::RuntimeScalar(0, false),
                Node::Add(2, 3),
                Node::Neg(4),
                Node::Sin(4),
            ],
            outputs: vec![4, 5, 6],
        };
        let Canonical { mapped, nodes } = canonical_nodes(&graph);
        let outputs: Vec<_> = graph.outputs.iter().map(|&id| mapped[id]).collect();
        let (units, _) = realization(&graph, &[Address::Linear], 257, &outputs, &nodes, false);
        assert_eq!(units.len(), 3);
        assert!(units.iter().all(|unit| unit.reads.is_empty()));
        assert!(units.iter().all(|unit| unit.shape.elements(257) == 257));
        assert_eq!(
            plan(&graph, &[Address::Linear], 257, &[0, 1, 2], false).len(),
            3
        );
    }

    #[test]
    fn original_producer_uses_its_real_planner_import() {
        let original = long_chain();
        let addresses = [Address::Linear, Address::Linear];
        let initial = super::plan(&original, &addresses, 2, &[0, 1], false);
        let candidates: BTreeSet<_> = initial
            .regions
            .iter()
            .flat_map(|region| region.imports.iter().copied())
            .collect();
        assert!(!candidates.is_empty());
        let mut checked = false;
        for representative in candidates {
            let mut graph = original.clone();
            let consumer = graph.nodes.len();
            graph.nodes.push(Node::Neg(representative));
            graph.outputs.push(consumer);
            let selected = super::plan(&graph, &addresses, 2, &[0, 1, 2], false);
            let canonical = selected.canonical.mapped[representative];
            let Some(region) = selected.regions.iter().find(|region| {
                region.imports.contains(&canonical) && region.roots.contains(&consumer)
            }) else {
                continue;
            };
            let lowered = super::super::lowering::region(
                &graph,
                &selected.canonical,
                &region.roots,
                &region.imports,
            );
            let output = region
                .roots
                .iter()
                .position(|&root| root == consumer)
                .unwrap();
            let mut value = lowered.roots[output];
            loop {
                use super::super::program::Operation;
                match lowered.operations[value] {
                    Operation::Copy(input) | Operation::Neg(input) | Operation::Flip(input) => {
                        value = input;
                    }
                    Operation::Import(imported) => {
                        assert_eq!(imported, canonical);
                        break;
                    }
                    ref other => {
                        panic!("original producer bypassed its realized import: {other:?}")
                    }
                }
            }
            checked = true;
            break;
        }
        assert!(
            checked,
            "expected a producer's consumer beyond an actual realized boundary"
        );
    }

    #[test]
    fn independent_tensor_producers_keep_distinct_dependencies() {
        let mut graph = Graph {
            inputs: 2,
            nodes: vec![
                Node::Input(0),
                Node::Input(1),
                Node::Mul(0, 1),
                Node::Mul(0, 1),
                Node::Sub(3, 0),
                Node::Sin(2),
            ],
            outputs: vec![],
        };
        // The distinction is required for both returned and internal tensors.
        // It does not prevent scalar expression sharing in a fused region.
        for outputs in [vec![4, 5], vec![2, 4, 5], vec![2, 3]] {
            graph.outputs = outputs;
            let canonical = canonical_nodes(&graph);
            assert_eq!(canonical.mapped, (0..graph.nodes.len()).collect::<Vec<_>>());
            assert_eq!(canonical.nodes[4], Node::Sub(3, 0));
            assert_eq!(canonical.nodes[5], Node::Sin(2));
        }
    }

    #[test]
    fn identical_computations_keep_original_output_roots() {
        let graph = Graph {
            inputs: 1,
            nodes: vec![Node::Input(0), Node::Sin(0), Node::Sin(0)],
            outputs: vec![1, 2],
        };
        assert_eq!(
            plan(&graph, &[Address::Linear], 1, &[1, 0], false),
            vec![
                Region {
                    roots: vec![2],
                    imports: vec![]
                },
                Region {
                    roots: vec![1],
                    imports: vec![]
                }
            ]
        );
        // Enough shared bytes permits horizontal fusion, not graph-level CSE.
        assert_eq!(
            plan(&graph, &[Address::Linear], 3, &[1, 0], false),
            vec![Region {
                roots: vec![2, 1],
                imports: vec![]
            }]
        );
    }
}
