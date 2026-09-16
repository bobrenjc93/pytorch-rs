//! Exercise the shared completion owner with a real selected direct executable.
use super::*;
use crate::pointwise_ir::{Graph, Node, indexing::Address, program::Program};
use std::{
    cell::{Cell, RefCell},
    sync::{Arc, Weak},
};

#[test]
#[allow(clippy::too_many_lines)] // Keep actual launch and injected completion in one owner audit.
fn direct_launch_failures_complete_before_releasing_inputs_kernel_scratch_and_outputs() {
    if device_count() == 0 {
        eprintln!("skipping direct completion ownership: CUDA unavailable");
        return;
    }
    let input = Arc::new(CudaFloat32Storage::from_host(&[1., -2., 3.], 0).unwrap());
    let graph = Graph {
        inputs: 1,
        nodes: vec![Node::Input(0), Node::Neg(0)],
        outputs: vec![1],
    };
    let addresses = vec![Address::Linear];
    let program = Program::build(&graph, &addresses, 3, &[0], false).unwrap();
    let source = program.direct_source(&graph, &addresses).unwrap().unwrap();
    let identity = jit::ExecutableIdentity::new(
        &graph,
        &addresses,
        0,
        jit::Kernel::checked_context(0).unwrap(),
        Some(&program),
    );
    let compilation = crate::pointwise_ir::program::Compilation { source, erf: false };
    let kernel =
        jit::Kernel::compile_selected(&graph, 0, addresses, compilation, identity).unwrap();
    let plan = PointwisePlan::new(&kernel, 3, &program).unwrap();
    let input_owner = Arc::downgrade(&input);
    let kernel_owner = Arc::downgrade(&kernel);
    let runtime = runtime().unwrap();
    for phase in ["launch", "completion", "both"] {
        let allocated = RefCell::new(Vec::new());
        let scratch_owner: RefCell<Option<Weak<CudaFloat32Storage>>> = RefCell::new(None);
        let completions = Cell::new(0);
        let error = TensorError::CudaRuntimeError {
            operation: "direct completion ownership test",
            message: phase.into(),
        };
        let assert_live = || {
            assert_eq!(input_owner.strong_count(), 1);
            assert_eq!(kernel_owner.strong_count(), 1);
            assert_eq!(scratch_owner.borrow().as_ref().unwrap().strong_count(), 1);
            let cache = CACHE.lock().unwrap();
            for pointer in allocated.borrow().iter() {
                assert!(!cache.iter().any(|entry| entry.2 == *pointer));
            }
        };
        let result = input.pointwise_outputs(
            3,
            1,
            || {
                let (scratch, _guard) =
                    CudaFloat32Storage::allocate(plan.layout.scratch_elements, 0)?;
                let scratch = Arc::new(scratch);
                *scratch_owner.borrow_mut() = Some(Arc::downgrade(&scratch));
                Ok(scratch)
            },
            || {
                let (output, _guard) = CudaFloat32Storage::allocate(3, 0)?;
                allocated.borrow_mut().push(output.data_ptr);
                Ok(output)
            },
            |scratch, outputs| {
                assert_live();
                // SAFETY: checked three-element allocations; all owners remain
                // live until the shared completion closure has synchronized.
                unsafe {
                    kernel.launch(
                        input.data_ptr as u64,
                        input.data_ptr as u64,
                        &[outputs[0].data_ptr as u64],
                        3,
                        plan.instructions.as_ref().unwrap().data_ptr as u64,
                        plan.instruction_count as u64,
                        scratch.data_ptr as u64,
                        plan.layout,
                        &[],
                    )?;
                }
                if phase == "launch" || phase == "both" {
                    Err(error.clone())
                } else {
                    Ok(())
                }
            },
            |_scratch, _outputs| {
                assert_live();
                completions.set(completions.get() + 1);
                // SAFETY: synchronize the same per-thread stream used by launch.
                runtime.check(
                    unsafe { (runtime.stream_synchronize)(std::ptr::without_provenance_mut(1)) },
                    "cudaStreamSynchronize",
                )?;
                if phase == "completion" || phase == "both" {
                    Err(error.clone())
                } else {
                    Ok(())
                }
            },
        );
        assert!(matches!(result, Err(actual) if actual == error));
        assert_eq!(completions.get(), 1);
        assert!(scratch_owner.borrow().as_ref().unwrap().upgrade().is_none());
        assert_eq!(input.copy_range(0, 3).unwrap(), [1., -2., 3.]);
    }
    drop((input, kernel));
    assert!(input_owner.upgrade().is_none());
    assert!(kernel_owner.upgrade().is_none());
}
