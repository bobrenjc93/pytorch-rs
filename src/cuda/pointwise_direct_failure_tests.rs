//! Exercise the shared completion owner with a real selected direct executable.
use super::*;
use crate::pointwise_ir::{Graph, Node, indexing::Address, program::Program};
use std::{
    cell::{Cell, RefCell},
    sync::Arc,
};

fn kernels() -> (Arc<jit::Kernel>, Arc<jit::Kernel>, Program) {
    let graph = Graph {
        inputs: 1,
        nodes: vec![Node::Input(0), Node::Neg(0), Node::Relu(0)],
        outputs: vec![1, 2],
    };
    let addresses = vec![Address::Linear];
    let program = Program::build(&graph, &addresses, 3, &[0, 1], false).unwrap();
    let source = program.direct_source(&graph, &addresses).unwrap().unwrap();
    let identity = jit::ExecutableIdentity::new(
        &graph,
        &addresses,
        0,
        jit::Kernel::checked_context(0).unwrap(),
        Some(&program),
    );
    let direct = jit::Kernel::compile_selected(&graph, 0, addresses, source, identity).unwrap();
    let vm = jit::Kernel::compile(&graph, 0).unwrap();
    (direct, vm, program)
}

#[test]
fn direct_resources_are_absent_while_vm_and_empty_accounting_remain_exact() {
    if device_count() == 0 {
        eprintln!("skipping direct/VM resource accounting: CUDA unavailable");
        return;
    }
    let input = CudaFloat32Storage::from_host(&[1., -2., 3.], 0).unwrap();
    let (direct, vm, program) = kernels();
    for kernel in [&direct, &vm] {
        let is_direct = kernel.identity.as_ref().is_some_and(|id| id.direct);
        for elements in [0, 3] {
            let uploads = POINTWISE_UPLOADS.get();
            let allocations = STORAGE_ALLOCATIONS.get();
            let plan = PointwisePlan::new(kernel, elements, &program).unwrap();
            let vm_buffer = usize::from(elements != 0 && !is_direct);
            assert_eq!(POINTWISE_UPLOADS.get(), uploads + vm_buffer);
            assert_eq!(STORAGE_ALLOCATIONS.get(), allocations + vm_buffer);
            assert_eq!(plan.instructions.is_some(), vm_buffer != 0);
            assert_eq!(plan.instruction_count, program.instruction_count());
            assert_eq!(plan.register_count, program.register_count());
            assert_eq!(
                plan.retained_heap_bytes(),
                plan.instructions
                    .as_ref()
                    .map_or(0, |data| data.allocation_bytes)
            );
            let layout = jit::LaunchLayout::new(elements, program.register_count()).unwrap();
            assert_eq!(plan.layout.blocks, layout.blocks);
            assert_eq!(plan.layout.threads, layout.threads);
            assert_eq!(plan.layout.workers, layout.workers);
            assert_eq!(plan.layout.scratch_elements, layout.scratch_elements);
            let before = STORAGE_ALLOCATIONS.get();
            let outputs = input
                .pointwise_jit(
                    &input,
                    if elements == 0 {
                        [usize::MAX; 2]
                    } else {
                        [0; 2]
                    },
                    elements,
                    [elements; 2],
                    kernel,
                    &[],
                    &plan,
                )
                .unwrap();
            assert_eq!(STORAGE_ALLOCATIONS.get(), before + 2 + vm_buffer);
            assert_eq!(POINTWISE_UPLOADS.get(), uploads + vm_buffer);
            assert_eq!(outputs.len(), 2);
            if elements == 0 {
                assert!(outputs.iter().all(|output| output.elements == 0));
            } else {
                assert_eq!(outputs[0].copy_range(0, 3).unwrap(), [-1., 2., -3.]);
                assert_eq!(outputs[1].copy_range(0, 3).unwrap(), [1., 0., 3.]);
                assert_ne!(outputs[0].data_ptr, outputs[1].data_ptr);
                assert!(
                    outputs
                        .iter()
                        .all(|output| output.data_ptr != input.data_ptr)
                );
            }
        }
    }
}

#[test]
#[allow(clippy::too_many_lines)] // Keep actual launch and injected completion in one owner audit.
fn direct_launch_failures_complete_before_releasing_inputs_kernel_and_outputs() {
    if device_count() == 0 {
        eprintln!("skipping direct completion ownership: CUDA unavailable");
        return;
    }
    let input = Arc::new(CudaFloat32Storage::from_host(&[1., -2., 3.], 0).unwrap());
    let (kernel, _vm, program) = kernels();
    let plan = PointwisePlan::new(&kernel, 3, &program).unwrap();
    assert!(plan.instructions.is_none());
    let input_owner = Arc::downgrade(&input);
    let kernel_owner = Arc::downgrade(&kernel);
    let runtime = runtime().unwrap();
    let current = usize::from(
        std::env::var("CUDA_VISIBLE_DEVICES").as_deref() == Ok("0,1") && device_count() >= 2,
    );
    let _guard = runtime.guard(current).unwrap();
    for phase in ["allocation", "launch", "completion", "both", "success"] {
        let allocated = RefCell::new(Vec::new());
        let launches = Cell::new(0);
        let completions = Cell::new(0);
        let error = |message: &str| TensorError::CudaRuntimeError {
            operation: "direct completion ownership test",
            message: message.into(),
        };
        let healthy = CACHE_HEALTHY.load(Ordering::Relaxed);
        let allocations = STORAGE_ALLOCATIONS.get();
        let assert_live = || {
            assert_eq!(input_owner.strong_count(), 1);
            assert_eq!(kernel_owner.strong_count(), 1);
            let cache = CACHE.lock().unwrap();
            for pointer in allocated.borrow().iter() {
                assert!(!cache.iter().any(|entry| entry.2 == *pointer));
            }
        };
        let result = input.pointwise_outputs(
            3,
            2,
            || Ok(None::<CudaFloat32Storage>),
            || {
                if phase == "allocation" && allocated.borrow().len() == 1 {
                    assert_live();
                    return Err(error("allocation"));
                }
                let (output, _guard) = CudaFloat32Storage::allocate(3, 0)?;
                assert!(!allocated.borrow().contains(&output.data_ptr));
                allocated.borrow_mut().push(output.data_ptr);
                Ok(output)
            },
            |scratch, outputs| {
                assert!(scratch.is_none());
                launches.set(launches.get() + 1);
                assert_live();
                // SAFETY: checked allocations and zero unused ABI pointers;
                // actual input/module/output owners span shared completion.
                unsafe {
                    kernel.launch(
                        input.data_ptr as u64,
                        input.data_ptr as u64,
                        &outputs
                            .iter()
                            .map(|output| output.data_ptr as u64)
                            .collect::<Vec<_>>(),
                        3,
                        0,
                        plan.instruction_count as u64,
                        0,
                        plan.layout,
                        &[],
                    )?;
                }
                if phase == "launch" || phase == "both" {
                    Err(error("launch"))
                } else {
                    Ok(())
                }
            },
            |scratch, _outputs| {
                assert!(scratch.is_none());
                assert_live();
                completions.set(completions.get() + 1);
                // SAFETY: synchronize the same legacy stream used by launch.
                runtime.check(
                    unsafe { (runtime.stream_synchronize)(std::ptr::without_provenance_mut(1)) },
                    "cudaStreamSynchronize",
                )?;
                if phase == "completion" || phase == "both" {
                    Err(error("completion"))
                } else {
                    Ok(())
                }
            },
        );
        let executed = usize::from(phase != "allocation");
        assert_eq!(launches.get(), executed);
        assert_eq!(completions.get(), executed);
        assert_eq!(
            STORAGE_ALLOCATIONS.get(),
            allocations + allocated.borrow().len()
        );
        if phase == "success" {
            let outputs = result.unwrap();
            assert_eq!(outputs[0].copy_range(0, 3).unwrap(), [-1., 2., -3.]);
            assert_eq!(outputs[1].copy_range(0, 3).unwrap(), [1., 0., 3.]);
        } else {
            let expected = if phase == "both" { "launch" } else { phase };
            assert!(matches!(result, Err(actual) if actual == error(expected)));
            if executed != 0 {
                assert!(!CACHE_HEALTHY.load(Ordering::Relaxed));
                let cache = CACHE.lock().unwrap();
                for pointer in allocated.borrow().iter() {
                    assert!(!cache.iter().any(|entry| entry.2 == *pointer));
                }
            } else {
                assert_eq!(CACHE_HEALTHY.load(Ordering::Relaxed), healthy);
            }
        }
        let mut restored = -1;
        runtime
            .check(
                unsafe { (runtime.get_device)(&raw mut restored) },
                "cudaGetDevice",
            )
            .unwrap();
        assert_eq!(usize::try_from(restored).unwrap(), current);
        assert_eq!(input.copy_range(0, 3).unwrap(), [1., -2., 3.]);
    }
    drop((input, kernel));
    assert!(input_owner.upgrade().is_none());
    assert!(kernel_owner.upgrade().is_none());
}
