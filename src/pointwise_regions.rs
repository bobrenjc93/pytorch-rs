//! Numerical regions inside one launch. These are not allocation or cache units.
use super::{Graph, Node, indexing::Address};
use std::collections::HashMap;

type Groups = Vec<Vec<usize>>;

fn operands(node: &Node) -> Vec<usize> {
    match *node {
        Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) => vec![a, b],
        Node::Neg(a) | Node::Relu(a) | Node::Sin(a) | Node::Cos(a) => vec![a],
        _ => vec![],
    }
}

fn merge(groups: &mut Groups, a: usize, b: usize) {
    let tail = groups.remove(b);
    groups[a].extend(tail);
    groups[a].sort_unstable();
}

struct ReadFacts {
    inputs: Vec<u8>,
    dependencies: Vec<u64>,
    canonical: Vec<usize>,
}

fn read_dependencies(graph: &Graph) -> Option<ReadFacts> {
    let mut live = vec![false; graph.nodes.len()];
    for &id in &graph.outputs {
        live[id] = true;
    }
    for id in (0..graph.nodes.len()).rev() {
        if live[id] {
            for a in operands(&graph.nodes[id]) {
                live[a] = true;
            }
        }
    }
    // Certify a simple realization regime, rather than imitating the general
    // Inductor scheduler: <= 3 scalar operations per original SSA node stays
    // below its >30-op realization threshold. Larger graphs remain admitted
    // and use joint lowering, including the same call-frontier rules.
    if live.iter().filter(|&&v| v).count() > 10
        || !graph
            .nodes
            .iter()
            .zip(&live)
            .any(|(node, &live)| live && matches!(node, Node::Sin(_) | Node::Cos(_)))
    {
        return None;
    }
    let (aliases, zeros) = super::lowering::early_aliases(graph);
    let mut inputs = vec![0u8; graph.nodes.len()];
    let mut reads = vec![0u128; graph.nodes.len()];
    let mut dependencies = vec![0u64; graph.nodes.len()];
    let mut canonical = Vec::new();
    let mut interned = HashMap::new();
    for (id, node) in graph.nodes.iter().enumerate() {
        let args = operands(node);
        let key = super::lowering::remap(node, &canonical);
        let next = interned.len();
        canonical.push(*interned.entry(key).or_insert(next));
        if aliases[id] != id {
            inputs[id] = inputs[aliases[id]];
            reads[id] = reads[aliases[id]];
            dependencies[id] = dependencies[aliases[id]];
        } else if !zeros[id] {
            if let Node::Input(i) = *node {
                inputs[id] = 1 << i;
                reads[id] = 1 << i;
            }
            for a in args {
                inputs[id] |= inputs[a];
                reads[id] |= reads[a];
                dependencies[id] |= dependencies[a];
            }
        }
        // Count actual read boundaries, including realized returned producers.
        if live[id] && reads[id].count_ones() > 4 {
            return None;
        }
        if let Ok(slot) = graph.outputs.binary_search(&id) {
            // Zero-buffer-read outputs (including runtime scalar expressions)
            // stay cheap; other returned tensors are realization boundaries.
            if inputs[id] != 0 {
                dependencies[id] |= 1 << slot;
                reads[id] = 1 << (graph.inputs + slot);
            }
        }
    }
    Some(ReadFacts {
        inputs,
        dependencies,
        canonical,
    })
}

pub(super) fn plans(graph: &Graph, addresses: &[Address]) -> Vec<(usize, Groups)> {
    let whole = vec![(1, vec![(0..graph.outputs.len()).collect()])];
    if graph.outputs.len() == 1 || addresses.iter().any(|a| *a != Address::Linear) {
        return whole;
    }
    let Some(ReadFacts {
        inputs,
        dependencies,
        canonical,
    }) = read_dependencies(graph)
    else {
        return whole;
    };
    let mut vertical: Groups = (0..graph.outputs.len()).map(|s| vec![s]).collect();
    loop {
        let pair = (0..vertical.len()).find_map(|a| {
            (a + 1..vertical.len()).find_map(|b| {
                vertical[a]
                    .iter()
                    .any(|&x| {
                        vertical[b].iter().any(|&y| {
                            let (rx, ry) = (graph.outputs[x], graph.outputs[y]);
                            canonical[rx] == canonical[ry]
                                || dependencies[rx] & (1 << y) != 0
                                || dependencies[ry] & (1 << x) != 0
                        })
                    })
                    .then_some((a, b))
            })
        });
        let Some((a, b)) = pair else {
            break;
        };
        merge(&mut vertical, a, b);
    }
    // Stock default compilation requires ten shared bytes for horizontal
    // fusion. Linear float32 reads contribute four bytes per distinct input
    // per element. Derive intervals; never specialize on a particular shape.
    let mut boundaries = vec![1];
    for count in 1..=graph.inputs {
        boundaries.push(10usize.div_ceil(4 * count));
    }
    boundaries.sort_unstable();
    boundaries.dedup();
    let mut result: Vec<(usize, Groups)> = Vec::new();
    for n in boundaries {
        let mut groups = vertical.clone();
        loop {
            let masks: Vec<_> = groups
                .iter()
                .map(|g| g.iter().fold(0u8, |m, &s| m | inputs[graph.outputs[s]]))
                .collect();
            let pair = (0..groups.len()).find_map(|a| {
                (a + 1..groups.len()).find_map(|b| {
                    ((masks[a] & masks[b]).count_ones() as usize * 4 * n >= 10).then_some((a, b))
                })
            });
            let Some((a, b)) = pair else {
                break;
            };
            merge(&mut groups, a, b);
        }
        if result.last().is_none_or(|(_, prior)| *prior != groups) {
            result.push((n, groups));
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn horizontal_threshold_counts_unique_reads() {
        let addresses = [Address::Linear, Address::Linear];
        for (one_input, boundary) in [(false, 2), (true, 3)] {
            let graph = nonlinear(one_input);
            assert_eq!(
                plans(&graph, &addresses),
                vec![(1, vec![vec![0], vec![1]]), (boundary, vec![vec![0, 1]])]
            );
            let source = graph.source().unwrap();
            assert_eq!(source.matches("__global__ void").count(), 1);
            assert!(source.contains(&format!("if (n < {boundary}ull)")));
        }
    }

    #[test]
    fn returned_runtime_producer_connects_vertical_consumers() {
        let mut graph = nonlinear(false);
        graph.outputs.insert(0, 2);
        assert_eq!(
            plans(&graph, &[Address::Linear, Address::Linear]),
            vec![(1, vec![vec![0, 1, 2]])]
        );
    }

    #[test]
    fn zero_read_runtime_producer_stays_cheap() {
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
        assert_eq!(
            plans(&graph, &[Address::Linear]),
            vec![(1, vec![vec![0], vec![1], vec![2]])]
        );
        assert!(graph.source().is_ok());
    }

    #[test]
    fn returned_read_budget_falls_back_without_rejecting_graph() {
        let graph = Graph {
            inputs: 1,
            nodes: vec![
                Node::Input(0),
                Node::Neg(0),
                Node::Sin(0),
                Node::Cos(0),
                Node::Relu(0),
                Node::Neg(0),
                Node::Add(1, 2),
                Node::Add(6, 3),
                Node::Add(7, 4),
                Node::Add(8, 5),
            ],
            // The final sum reads five returned producers. Its partial sums
            // are deliberately not returned (which would reset their reads).
            outputs: vec![1, 2, 3, 4, 5, 9],
        };
        assert!(read_dependencies(&graph).is_none());
        assert_eq!(
            plans(&graph, &[Address::Linear]),
            vec![(1, vec![(0..6).collect()])]
        );
        assert!(graph.source().is_ok());
    }

    #[test]
    fn larger_and_broadcast_graphs_remain_admitted() {
        let mut graph = nonlinear(false);
        for _ in 0..7 {
            graph.nodes.push(Node::Sin(graph.nodes.len() - 1));
        }
        graph.outputs[1] = graph.nodes.len() - 1;
        assert_eq!(
            plans(&graph, &[Address::Linear, Address::Linear]),
            vec![(1, vec![vec![0, 1]])]
        );
        assert!(graph.source().is_ok());
        let graph = nonlinear(false);
        assert_eq!(
            plans(&graph, &[Address::Linear, Address::Broadcast(vec![])]),
            vec![(1, vec![vec![0, 1]])]
        );
    }
}
