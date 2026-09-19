use super::*;
use crate::pointwise_ir::{Graph, Node};

fn cuda_input(values: Vec<f32>, shape: &[usize]) -> Tensor {
    Tensor::from_vec(values, shape)
        .unwrap()
        .try_copy_cpu_to_cuda(Device::Cuda(0))
        .unwrap()
}

fn available() -> bool {
    if crate::cuda::device_count() == 0 {
        eprintln!("skipping native Program identity test: CUDA unavailable");
        false
    } else {
        true
    }
}

fn scalar_graph() -> Graph {
    Graph {
        inputs: 1,
        nodes: vec![
            Node::Input(0),
            Node::RuntimeScalar(0, false),
            Node::Mul(0, 1),
        ],
        outputs: vec![2],
    }
}

#[test]
fn equivalent_host_preparations_reuse_one_kernel_and_release_construction_owners() {
    if !available() {
        return;
    }
    let graph = scalar_graph();
    let first = cuda_input(vec![1.; 3], &[3]);
    let builds = crate::pointwise_ir::program::build_count();
    let uploads = crate::cuda::pointwise_upload_count();
    let host = HostPointwise::new(&[&first], graph.clone(), None, &[0]).unwrap();
    assert!(host.identity.direct);
    assert_eq!(crate::pointwise_ir::program::build_count(), builds + 1);
    assert_eq!(crate::cuda::pointwise_upload_count(), uploads);
    let kernel = host.compile().unwrap();
    let owner = Arc::downgrade(&kernel);
    let identity = host.identity.clone();
    drop(host);
    let mut retained = Vec::new();
    for size in [3, 5, 7, 9, 5] {
        let input = cuda_input(vec![2.; size], &[size]);
        let input_owner = Arc::downgrade(&input.storage);
        let host = HostPointwise::new(&[&input], graph.clone(), None, &[0]).unwrap();
        assert_eq!(host.identity, identity);
        let prepared = host.bind(Arc::clone(&kernel)).unwrap();
        assert!(Arc::ptr_eq(prepared.kernel(), &kernel));
        assert_eq!(
            prepared.instruction_count(),
            host.program.instruction_count()
        );
        assert_eq!(prepared.register_count(), host.program.register_count());
        drop(host);
        assert_eq!(crate::cuda::pointwise_upload_count(), uploads);
        let before = crate::pointwise_ir::program::build_count();
        let first = prepared.run(&[&input], &[3.]).unwrap().remove(0);
        let second = prepared.run(&[&input], &[-2.]).unwrap().remove(0);
        assert_eq!(
            first.try_copy_cuda_to_cpu().unwrap().try_to_vec().unwrap(),
            vec![6.; size]
        );
        assert_eq!(
            second.try_copy_cuda_to_cpu().unwrap().try_to_vec().unwrap(),
            vec![-4.; size]
        );
        assert!(!first.shares_storage_with(&second));
        assert!(!first.shares_storage_with(&input));
        assert_eq!(crate::pointwise_ir::program::build_count(), before);
        assert_eq!(crate::cuda::pointwise_upload_count(), uploads);
        retained.push(prepared);
        if retained.len() > 2 {
            retained.remove(0);
        }
        drop(input);
        assert!(input_owner.upgrade().is_none());
    }
    drop(kernel);
    assert_eq!(owner.strong_count(), 2);
    drop(retained);
    assert!(owner.upgrade().is_none());
}

#[test]
fn bind_rejects_mismatched_identity_before_upload_and_run_checks_exact_shape() {
    if !available() {
        return;
    }
    let input = cuda_input(vec![1.; 3], &[3]);
    let host = HostPointwise::new(&[&input], scalar_graph(), None, &[0]).unwrap();
    let kernel = host.compile().unwrap();
    let uploads = crate::cuda::pointwise_upload_count();
    let mut other_graph = scalar_graph();
    other_graph.nodes.push(Node::Neg(0));
    let different = HostPointwise::new(&[&input], other_graph, None, &[0]).unwrap();
    assert!(different.bind(Arc::clone(&kernel)).is_err());
    assert_eq!(crate::cuda::pointwise_upload_count(), uploads);
    let legacy = Kernel::compile(&scalar_graph(), 0).unwrap();
    assert!(host.bind(legacy).is_err());
    assert_eq!(crate::cuda::pointwise_upload_count(), uploads);
    let prepared = host.bind(kernel).unwrap();
    let mut wrong_shape = cuda_input(vec![1.; 3], &[3]);
    wrong_shape.shape = vec![1, 3];
    wrong_shape.strides = vec![3, 1];
    assert!(prepared.run(&[&wrong_shape], &[2.]).is_err());
    assert!(prepared.run(&[&input], &[]).is_err());
    assert_eq!(crate::cuda::pointwise_upload_count(), uploads);
}

#[test]
fn direct_and_legacy_vm_have_identical_float_bits_for_native_graphs() {
    if !available() {
        return;
    }
    let values = vec![
        0.,
        -0.,
        f32::from_bits(1),
        -f32::from_bits(1),
        0.125,
        -0.25,
        2.,
        -3.,
    ];
    let input = cuda_input(values, &[8]);
    let graph = Graph {
        inputs: 1,
        nodes: vec![
            Node::Input(0),
            Node::RuntimeScalar(0, false),
            Node::Mul(0, 1),
            Node::Add(2, 0),
            Node::Sub(0, 2),
            Node::Neg(0),
            Node::Relu(0),
            Node::Sin(0),
            Node::Cos(0),
            Node::Constant(0.25f64.to_bits()),
            Node::Constant(0.0f64.to_bits()),
            Node::Mul(0, 10),
            Node::Add(11, 9),
            Node::Sin(12),
            Node::Cos(12),
        ],
        outputs: vec![3, 4, 5, 6, 7, 8, 13, 14],
    };
    let order: Vec<_> = (0..graph.outputs.len()).rev().collect();
    let host = HostPointwise::new(&[&input], graph.clone(), None, &order).unwrap();
    assert!(host.identity.direct);
    let direct = host.bind(host.compile().unwrap()).unwrap();
    let vm = Tensor::prepare_pointwise(
        &[&input],
        Kernel::compile(&graph, 0).unwrap(),
        None,
        Some(&order),
    )
    .unwrap();
    for scalar in [2., -2., -0.] {
        let direct_outputs = direct.run(&[&input], &[scalar]).unwrap();
        let vm_outputs = vm.run(&[&input], &[scalar]).unwrap();
        for (a, b) in direct_outputs.iter().zip(&vm_outputs) {
            let bits = |tensor: &Tensor| {
                tensor
                    .try_copy_cuda_to_cpu()
                    .unwrap()
                    .try_to_vec()
                    .unwrap()
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>()
            };
            assert_eq!(bits(a), bits(b));
            assert!(!a.shares_storage_with(b));
        }
    }
}

#[test]
fn empty_outputs_select_vm_and_scalar_outputs_select_direct() {
    if !available() {
        return;
    }
    let empty = cuda_input(vec![], &[0]);
    let host = HostPointwise::new(&[&empty], scalar_graph(), None, &[0]).unwrap();
    assert!(!host.identity.direct);
    let prepared = host.bind(host.compile().unwrap()).unwrap();
    assert_eq!(prepared.run(&[&empty], &[2.]).unwrap()[0].shape(), &[0]);
    let scalar = cuda_input(vec![3.], &[]);
    let host = HostPointwise::new(&[&scalar], scalar_graph(), None, &[0]).unwrap();
    assert!(host.identity.direct);
    let output = host
        .bind(host.compile().unwrap())
        .unwrap()
        .run(&[&scalar], &[2.])
        .unwrap()
        .remove(0);
    assert!(output.shape().is_empty());
    assert_eq!(
        output.try_copy_cuda_to_cpu().unwrap().try_to_vec().unwrap(),
        [6.]
    );
}

#[test]
fn actual_broadcast_addresses_cannot_bind_a_linear_executor() {
    if !available() {
        return;
    }
    let graph = Graph {
        inputs: 2,
        nodes: vec![Node::Input(0), Node::Input(1), Node::Add(0, 1)],
        outputs: vec![2],
    };
    let vector = cuda_input(vec![1., 2., 3.], &[3]);
    let scalar = cuda_input(vec![10.], &[1]);
    let linear = HostPointwise::new(&[&vector, &vector], graph.clone(), None, &[0]).unwrap();
    let broadcast = HostPointwise::new(&[&vector, &scalar], graph, None, &[0]).unwrap();
    assert_ne!(linear.addresses, broadcast.addresses);
    assert_ne!(linear.identity, broadcast.identity);
    let kernel = linear.compile().unwrap();
    let uploads = crate::cuda::pointwise_upload_count();
    assert!(broadcast.bind(kernel).is_err());
    assert_eq!(crate::cuda::pointwise_upload_count(), uploads);
    let output = broadcast
        .bind(broadcast.compile().unwrap())
        .unwrap()
        .run(&[&vector, &scalar], &[])
        .unwrap()
        .remove(0);
    assert_eq!(
        output.try_copy_cuda_to_cpu().unwrap().try_to_vec().unwrap(),
        [11., 12., 13.]
    );
}

#[test]
fn over_cap_host_programs_use_one_vm_identity_and_keep_independent_uploads() {
    if !available() {
        return;
    }
    let graph = Graph {
        inputs: 1,
        nodes: std::iter::once(Node::Input(0))
            .chain((0..260).map(Node::Sin))
            .collect(),
        outputs: vec![260],
    };
    let first = cuda_input(vec![0.; 3], &[3]);
    let second = cuda_input(vec![0.; 5], &[5]);
    let a = HostPointwise::new(&[&first], graph.clone(), None, &[0]).unwrap();
    let b = HostPointwise::new(&[&second], graph, None, &[0]).unwrap();
    assert!(a.program.instruction_count() > 256);
    assert!(!a.identity.direct);
    assert_eq!(a.identity, b.identity);
    let kernel = a.compile().unwrap();
    let uploads = crate::cuda::pointwise_upload_count();
    let a = a.bind(Arc::clone(&kernel)).unwrap();
    let b = b.bind(kernel).unwrap();
    assert_eq!(crate::cuda::pointwise_upload_count(), uploads + 2);
    for (prepared, input, size) in [(&a, &first, 3), (&b, &second, 5)] {
        let output = prepared.run(&[input], &[]).unwrap().remove(0);
        assert_eq!(
            output.try_copy_cuda_to_cpu().unwrap().try_to_vec().unwrap(),
            vec![0.; size]
        );
    }
}

fn read_vm_outputs(outputs: &[Tensor]) -> Vec<Vec<f32>> {
    outputs
        .iter()
        .map(|output| output.try_copy_cuda_to_cpu().unwrap().try_to_vec().unwrap())
        .collect()
}

fn assert_vm_output_bits(actual: &[Vec<f32>], expected: &[Vec<f32>]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert_eq!(actual.is_nan(), expected.is_nan());
            if !expected.is_nan() {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
    }
}

#[test]
fn different_uploaded_programs_share_only_the_vm_executable() {
    if !available() {
        return;
    }
    let mut nodes = vec![Node::Input(0), Node::Mul(0, 0), Node::Neg(1), Node::Sin(1)];
    nodes.extend((3..263).map(Node::Sin));
    let graph = Graph {
        inputs: 1,
        nodes,
        outputs: vec![2, 263],
    };
    // The underflow lane distinguishes fused negative product (-0) from
    // subtraction after positive-product realization (+0). Other lanes retain
    // ordinary values, both zero signs, subnormals and nonfinite masks/signs.
    let input = cuda_input(
        vec![
            1e-38,
            -1e-38,
            0.,
            -0.,
            f32::from_bits(1),
            -f32::from_bits(1),
            0.125,
            -0.125,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            1.,
            -1.,
        ],
        &[13],
    );
    let mut hosts = Vec::new();
    for hint in [1, 2, 13] {
        for order in [[0, 1], [1, 0]] {
            hosts.push((
                hint,
                order,
                HostPointwise::new(&[&input], graph.clone(), Some(hint), &order).unwrap(),
            ));
        }
    }
    assert!(hosts.iter().all(|(_, _, host)| !host.identity.direct));
    assert!(
        hosts
            .iter()
            .any(|(_, _, host)| host.program.instructions() != hosts[0].2.program.instructions())
    );
    let kernel = hosts[0].2.compile().unwrap();
    let legacy = Kernel::compile(&graph, 0).unwrap();
    assert!(!Arc::ptr_eq(&kernel, &legacy));
    let mut retained = Vec::new();
    let mut witness = Vec::new();
    for (hint, order, host) in &hosts {
        assert_eq!(host.identity, hosts[0].2.identity);
        let prepared = host.bind(Arc::clone(&kernel)).unwrap();
        assert!(Arc::ptr_eq(prepared.kernel(), &kernel));
        // Independently replan the corresponding legacy invocation; do not use
        // the first host's words or a regenerated diagnostic as the oracle.
        let expected =
            Tensor::prepare_pointwise(&[&input], Arc::clone(&legacy), Some(*hint), Some(order))
                .unwrap()
                .run(&[&input], &[])
                .unwrap();
        let expected = read_vm_outputs(&expected);
        let outputs = prepared.run(&[&input], &[]).unwrap();
        assert_eq!(outputs.len(), 2);
        assert!(!outputs[0].shares_storage_with(&outputs[1]));
        assert!(
            outputs
                .iter()
                .all(|output| !output.shares_storage_with(&input))
        );
        assert_vm_output_bits(&read_vm_outputs(&outputs), &expected);
        witness.push(expected[0][0].to_bits());
        retained.push((outputs, expected));
        for (earlier, expected) in &retained {
            assert_vm_output_bits(&read_vm_outputs(earlier), expected);
        }
    }
    // Demonstrate a numerical witness, not merely differing instruction words.
    assert_eq!(witness[0], (-0.0_f32).to_bits());
    assert_eq!(witness[4], 0.0_f32.to_bits());
    assert_ne!(witness[0], witness[4]);
}
