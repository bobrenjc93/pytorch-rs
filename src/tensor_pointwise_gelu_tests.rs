//! Same-Program execution controls: selection changes, numerical lowering does not.
use super::*;
use crate::pointwise_ir::{Graph, Node};

fn input(values: Vec<f32>) -> Tensor {
    let size = values.len();
    Tensor::from_vec(values, [size])
        .unwrap()
        .try_copy_cpu_to_cuda(Device::Cuda(0))
        .unwrap()
}

fn available() -> bool {
    if crate::cuda::device_count() == 0 {
        eprintln!("skipping selected GELU Programs: CUDA unavailable");
        return false;
    }
    true
}

fn graph() -> Graph {
    Graph {
        inputs: 1,
        nodes: vec![
            Node::Input(0),
            Node::Constant(0.5_f64.to_bits()),
            Node::Mul(0, 1),
            Node::Constant(std::f64::consts::FRAC_1_SQRT_2.to_bits()),
            Node::Mul(0, 3),
            Node::Erf(4),
            Node::Integer(1.0_f64.to_bits()),
            Node::Add(6, 5),
            Node::Mul(2, 7),
            Node::Erf(8),
            Node::RuntimeScalar(0, false),
            Node::Mul(9, 10),
            Node::Sub(11, 9),
            Node::Integer(0),
            Node::Mul(0, 13),
            Node::Constant(0.75_f64.to_bits()),
            Node::Add(14, 15),
            Node::Erf(16),
        ],
        outputs: vec![5, 8, 9, 12, 17],
    }
}

#[test]
fn selected_erf_direct_and_vm_execute_the_same_words_and_preserve_owners() {
    if !available() {
        return;
    }
    let values = vec![
        0.,
        -0.,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        -5.,
        -1.,
        0.375,
        5.,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7fc0_1234),
    ];
    let input = input(values);
    for hint in [0, 1, 13, u64::MAX] {
        for order in [vec![0, 1, 2, 3, 4], vec![4, 3, 2, 1, 0]] {
            let mut host = HostPointwise::new(&[&input], graph(), Some(hint), &order).unwrap();
            assert!(host.identity.direct);
            assert!(host.compilation.erf);
            let words = host.program.instructions().to_vec();
            let builds = crate::pointwise_ir::program::build_count();
            let kernel = host.compile().unwrap();
            let direct_owner = Arc::downgrade(&kernel);
            let direct = host.bind(kernel).unwrap();
            // Alter only the test's selected payload and identity. Reuse the
            // very same Program owner, register layout, words and launch ABI.
            host.compilation = host.graph.compilation(&host.addresses).unwrap();
            host.identity = crate::cuda::jit::ExecutableIdentity::new(
                &host.graph,
                &host.addresses,
                host.device,
                host.identity.context,
                None,
            );
            let kernel = host.compile().unwrap();
            let vm_owner = Arc::downgrade(&kernel);
            let vm = host.bind(kernel).unwrap();
            assert_eq!(words, host.program.instructions());
            assert_eq!(crate::pointwise_ir::program::build_count(), builds);
            for scalar in [1., -2., -0., f32::from_bits(1)] {
                let a = direct.run(&[&input], &[scalar]).unwrap();
                let b = vm.run(&[&input], &[scalar]).unwrap();
                for (slot, (a, b)) in a.iter().zip(&b).enumerate() {
                    let read = |t: &Tensor| t.try_copy_cuda_to_cpu().unwrap().try_to_vec().unwrap();
                    let av = read(a);
                    let bv = read(b);
                    for (index, (&x, &y)) in av.iter().zip(&bv).enumerate() {
                        if x.is_nan() || y.is_nan() {
                            // NaN payload/sign encoding is not a public guarantee.
                            assert!(x.is_nan() && y.is_nan());
                        } else {
                            assert_eq!(
                                x.to_bits(),
                                y.to_bits(),
                                "hint {hint} slot {slot} lane {index}"
                            );
                        }
                    }
                    eprintln!(
                        "same-Program hint={hint} order={order:?} scalar={:08x} slot={slot} words={words:?} direct={:?} vm={:?}",
                        scalar.to_bits(),
                        av.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        bv.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                    assert!(!a.shares_storage_with(b));
                    assert!(!a.shares_storage_with(&input));
                }
            }
            drop(host);
            assert_eq!(direct_owner.strong_count(), 1);
            assert_eq!(vm_owner.strong_count(), 1);
            drop((direct, vm));
            assert!(direct_owner.upgrade().is_none());
            assert!(vm_owner.upgrade().is_none());
        }
    }
}

#[test]
fn selected_empty_dead_erased_and_over_cap_erf_dependencies() {
    if !available() {
        return;
    }
    let mut graph = Graph {
        inputs: 1,
        nodes: vec![
            Node::Input(0),
            Node::Erf(0),
            Node::Integer(0),
            Node::Mul(1, 2),
            Node::Neg(0),
        ],
        outputs: vec![1],
    };
    for size in [0, 3] {
        let x = input(vec![0.; size]);
        for (output, erf) in [(4, false), (1, true), (3, false)] {
            graph.outputs = vec![output];
            let host = HostPointwise::new(&[&x], graph.clone(), None, &[0]).unwrap();
            assert_eq!(host.identity.direct, size != 0);
            assert_eq!(host.compilation.erf, erf);
            assert_eq!(host.program.uses_erf(), erf);
            let kernel = host.compile().unwrap();
            assert_eq!(
                kernel
                    .options
                    .contains(&"--relocatable-device-code=true".into()),
                erf
            );
            assert_eq!(kernel.source.contains("torch_rs_erf"), erf);
            let result = host.bind(kernel).unwrap().run(&[&x], &[]).unwrap();
            assert_eq!(result[0].shape(), &[size]);
        }
    }
    graph.nodes.truncate(2);
    graph.nodes.extend((1..261).map(Node::Sin));
    graph.outputs = vec![1, 261];
    let x = input(vec![0.125; 13]);
    let first = HostPointwise::new(&[&x], graph.clone(), Some(1), &[0, 1]).unwrap();
    assert!(!first.identity.direct);
    let kernel = first.compile().unwrap();
    for hint in [0, 1, 13, u64::MAX] {
        for order in [[0, 1], [1, 0]] {
            let host = HostPointwise::new(&[&x], graph.clone(), Some(hint), &order).unwrap();
            assert!(host.program.instruction_count() > 256);
            assert!(host.compilation.erf);
            assert_eq!(host.identity, first.identity);
            assert_eq!(host.compilation.source, first.compilation.source);
            assert_eq!(
                host.bind(Arc::clone(&kernel))
                    .unwrap()
                    .run(&[&x], &[])
                    .unwrap()
                    .len(),
                2
            );
        }
    }
}
