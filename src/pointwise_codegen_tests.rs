use super::*;
use crate::{cuda::jit::ExecutableIdentity, pointwise_ir::Node};

fn graph() -> Graph {
    // Enough original nodes for the validator's independent structural bounds.
    Graph {
        inputs: 1,
        nodes: std::iter::once(Node::Input(0))
            .chain(std::iter::repeat_n(Node::RuntimeScalar(0, false), 32))
            .collect(),
        outputs: vec![0],
    }
}

fn program(instructions: Vec<[u32; 6]>, registers: usize) -> Program {
    let program = Program {
        instructions,
        register_count: registers,
        listing: None,
    };
    program.validate(&graph()).unwrap();
    program
}

#[test]
fn exact_instruction_and_register_caps_are_inclusive() {
    assert_eq!(
        (MAX_INSTRUCTIONS, MAX_REGISTERS, MAX_SOURCE_BYTES),
        (256, 128, 65_536)
    );
    let mut words = vec![[INPUT, 0, 0, 0, 0, 0]];
    words.extend(std::iter::repeat_n([COPY, 0, 0, 0, 0, 0], 254));
    words.push([STORE, 0, 0, 0, 0, 0]);
    let mut program = program(words, 128);
    assert!(
        source(&program, &graph(), &[Address::Linear])
            .unwrap()
            .is_some()
    );
    program.register_count = 129;
    program.validate(&graph()).unwrap();
    assert!(
        source(&program, &graph(), &[Address::Linear])
            .unwrap()
            .is_none()
    );
    program.register_count = 128;
    program.instructions.insert(1, [COPY, 0, 0, 0, 0, 0]);
    program.validate(&graph()).unwrap();
    assert_eq!(program.instruction_count(), 257);
    assert!(
        source(&program, &graph(), &[Address::Linear])
            .unwrap()
            .is_none()
    );
}

#[test]
fn complete_source_limit_includes_abi_addresses_and_trailer() {
    let program = program(vec![[INPUT, 0, 0, 0, 0, 0], [STORE, 0, 0, 0, 0, 0]], 1);
    let linear = source(&program, &graph(), &[Address::Linear])
        .unwrap()
        .unwrap();
    let fixed_bytes = linear.len() - 1; // Replace exactly the linear `i` address.
    let mut terms = Vec::new();
    while fixed_bytes + Address::Broadcast(terms.clone()).source().len() < MAX_SOURCE_BYTES {
        terms.push((1, 1, 1));
    }
    terms.pop();
    let missing = MAX_SOURCE_BYTES - fixed_bytes - Address::Broadcast(terms.clone()).source().len();
    // Decimal widths in two address operands bridge the gap between terms.
    let first = missing.min(18);
    let second = missing - first;
    assert!(second <= 18);
    let last = terms.last_mut().unwrap();
    last.0 = 10usize.pow(u32::try_from(first).unwrap());
    last.1 = 10usize.pow(u32::try_from(second).unwrap());
    let address = Address::Broadcast(terms.clone());
    assert_eq!(fixed_bytes + address.source().len(), MAX_SOURCE_BYTES);
    let emitted = source(&program, &graph(), &[address]).unwrap().unwrap();
    assert_eq!(emitted.len(), MAX_SOURCE_BYTES);
    assert!(emitted.ends_with("}\n}\n"));
    // One more UTF-8 byte must select VM, without dropping address/ABI text.
    terms.last_mut().unwrap().2 = 10;
    assert_eq!(
        fixed_bytes + Address::Broadcast(terms.clone()).source().len(),
        MAX_SOURCE_BYTES + 1
    );
    assert!(
        source(&program, &graph(), &[Address::Broadcast(terms)])
            .unwrap()
            .is_none()
    );
    assert!(source(&program, &graph(), &[]).is_err());
}

#[test]
fn every_opcode_and_fma_flag_preserves_the_vm_expression() {
    let mut words = vec![
        [INPUT, 0, 0, 0, 0, 0],
        [SCALAR, 1, 0, 0, 0, 0],
        [SCALAR, 1, 0, 0, 0, 1],
        [CONSTANT, 2, 0x8000_0000, 0, 0, 0],
    ];
    for opcode in COPY..=CONSTANT_COS {
        words.push([opcode, 2, 0, 1, 0, 0]);
    }
    for flags in 0..8 {
        words.push([FMA, 2, 0, 1, 2, flags]);
    }
    words.push([STORE, 0, 2, 0, 0, 0]);
    let program = program(words, 3);
    let emitted = source(&program, &graph(), &[Address::Linear])
        .unwrap()
        .unwrap();
    for expression in [
        "r0 = x0[i];",
        "r1 = s0;",
        "r1 = __uint_as_float(__float_as_uint(s0) ^ 0x80000000u);",
        "r2 = __uint_as_float(0x80000000u);",
        "r2 = r0;",
        "r2 = __fadd_rn(r0, r1);",
        "r2 = __fsub_rn(r0, r1);",
        "r2 = __fmul_rn(r0, r1);",
        "r2 = __fsub_rn(0.0f, r0);",
        "r2 = __uint_as_float(__float_as_uint(r0) ^ 0x80000000u);",
        "r2 = (r0 < 0.0f ? 0.0f : r0);",
        "r2 = sinf((__float_as_uint(r0) & 0x7f800000u) == 0 ? __uint_as_float(__float_as_uint(r0) & 0x80000000u) : r0);",
        "r2 = cosf(r0);",
        "r2 = (float)sin((double)r0);",
        "r2 = (float)cos((double)r0);",
        "out0[i] = r2;",
    ] {
        assert!(emitted.contains(expression), "missing {expression}");
    }
    let products = ["r0", "__uint_as_float(__float_as_uint(r0) ^ 0x80000000u)"];
    let addends = [
        "r2",
        "__uint_as_float(__float_as_uint(r2) ^ 0x80000000u)",
        "0.0f",
        "__uint_as_float(__float_as_uint(0.0f) ^ 0x80000000u)",
    ];
    for product in products {
        for addend in addends {
            assert!(emitted.contains(&format!("r2 = fmaf({product}, r1, {addend});")));
        }
    }
    assert_eq!(emitted.matches("fmaf(").count(), 8);
    assert!(program.listing.is_none());
}

#[test]
fn identity_compares_exact_graph_address_program_register_domain_and_context() {
    let graph = graph();
    let mut program = program(vec![[INPUT, 0, 0, 0, 0, 0], [STORE, 0, 0, 0, 0, 0]], 1);
    let identity = |graph: &Graph, address: Address, device, context, program: Option<&Program>| {
        ExecutableIdentity::new(graph, &[address], device, context, program)
    };
    let direct = identity(&graph, Address::Linear, 0, 123, Some(&program));
    assert_eq!(
        direct,
        identity(&graph.clone(), Address::Linear, 0, 123, Some(&program))
    );
    let vm = identity(&graph, Address::Linear, 0, 123, None);
    assert_ne!(direct, vm);
    let mut changed_graph = graph.clone();
    changed_graph.nodes.push(Node::Neg(0)); // Dead original IR is part of identity.
    assert_ne!(
        direct,
        identity(&changed_graph, Address::Linear, 0, 123, Some(&program))
    );
    for address in [
        Address::Broadcast(vec![]),
        Address::Broadcast(vec![(1, 3, 1)]),
        Address::Broadcast(vec![(1, 5, 1)]),
    ] {
        assert_ne!(direct, identity(&graph, address, 0, 123, Some(&program)));
    }
    assert_ne!(
        direct,
        identity(&graph, Address::Linear, 1, 123, Some(&program))
    );
    assert_ne!(
        direct,
        identity(&graph, Address::Linear, 0, 124, Some(&program))
    );
    program.register_count = 2;
    assert_ne!(
        direct,
        identity(&graph, Address::Linear, 0, 123, Some(&program))
    );
    program.register_count = 1;
    program.instructions.insert(1, [FLIP, 0, 0, 0, 0, 0]);
    program.validate(&graph).unwrap();
    assert_ne!(
        direct,
        identity(&graph, Address::Linear, 0, 123, Some(&program))
    );
    assert_eq!(vm, identity(&graph, Address::Linear, 0, 123, None));
}
