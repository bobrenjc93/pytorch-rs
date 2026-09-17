//! Untimed allocation and completion witnesses on the canonical JIT invocation.
use super::*;
use crate::pointwise_ir::{Graph, Node, indexing::Address, program::Program};
use std::{
    cell::RefCell,
    sync::{Arc, Weak},
};

#[derive(Clone, Copy)]
pub(super) enum Event<'a> {
    Allocating,
    Allocated(usize),
    Launched(Option<&'a CudaFloat32Storage>, &'a [CudaFloat32Storage]),
    Completed(Option<&'a CudaFloat32Storage>, &'a [CudaFloat32Storage]),
}

struct Observation {
    phase: &'static str,
    scratch: bool,
    allocations: usize,
    pointers: Vec<usize>,
    dropped: Vec<usize>,
    launches: usize,
    completions: usize,
    protected: Vec<usize>,
    input: Weak<CudaFloat32Storage>,
    kernel: Weak<jit::Kernel>,
}

thread_local! {
    static OBSERVATION: RefCell<Option<Observation>> = const { RefCell::new(None) };
}

fn injected(phase: &'static str) -> TensorError {
    TensorError::CudaRuntimeError {
        operation: "canonical pointwise invocation test",
        message: phase.into(),
    }
}

pub(super) fn record_drop(pointer: usize) {
    OBSERVATION.with(|slot| {
        if let Some(state) = slot.borrow_mut().as_mut() {
            state.dropped.push(pointer);
        }
    });
}

pub(super) fn observe(event: Event<'_>) -> Result<(), TensorError> {
    OBSERVATION.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(state) = slot.as_mut() else {
            return Ok(());
        };
        match event {
            Event::Allocating => {
                state.allocations += 1;
                // Fail the second output, after an actual first output allocation.
                if state.phase == "allocation"
                    && state.allocations == 2 + usize::from(state.scratch)
                {
                    assert!(state.dropped.is_empty());
                    return Err(injected("allocation"));
                }
            }
            Event::Allocated(pointer) => state.pointers.push(pointer),
            Event::Launched(scratch, outputs) | Event::Completed(scratch, outputs) => {
                let completion = matches!(event, Event::Completed(..));
                assert_eq!(scratch.is_some(), state.scratch);
                assert!(state.dropped.is_empty());
                assert!(state.input.upgrade().is_some());
                assert!(state.kernel.upgrade().is_some());
                assert_eq!(outputs.len(), 2);
                let cache = CACHE.lock().unwrap();
                for pointer in state.pointers.iter().chain(&state.protected) {
                    assert!(!cache.iter().any(|entry| entry.2 == *pointer));
                }
                drop(cache);
                if let Some(scratch) = scratch {
                    assert_ne!(scratch.data_ptr, 0);
                    assert_eq!(state.pointers[0], scratch.data_ptr);
                }
                for output in outputs {
                    assert!(state.pointers.contains(&output.data_ptr));
                }
                if completion {
                    state.completions += 1;
                    assert_eq!(state.launches, 1);
                    // This injection is AFTER real synchronization. It is not
                    // evidence of a device that failed to finish its work.
                    if matches!(state.phase, "completion" | "both") {
                        return Err(injected("completion"));
                    }
                } else {
                    state.launches += 1;
                    if matches!(state.phase, "launch" | "both") {
                        return Err(injected("launch"));
                    }
                }
            }
        }
        Ok(())
    })
}

struct ResetObservation;
impl Drop for ResetObservation {
    fn drop(&mut self) {
        OBSERVATION.with(|slot| *slot.borrow_mut() = None);
    }
}

#[test]
#[allow(clippy::too_many_lines)] // One canonical-path owner audit, including all failure phases.
fn direct_launch_failures_complete_before_releasing_inputs_kernel_scratch_and_outputs() {
    if device_count() == 0 {
        eprintln!("MISSING COVERAGE: canonical scratch ownership: CUDA unavailable");
        return;
    }
    let input = Arc::new(CudaFloat32Storage::from_host(&[1., -2., 3.], 0).unwrap());
    let graph = Graph {
        inputs: 1,
        nodes: vec![Node::Input(0), Node::Neg(0), Node::Neg(1)],
        outputs: vec![1, 2],
    };
    let addresses = vec![Address::Linear];
    let program = Program::build(&graph, &addresses, 3, &[0, 1], false).unwrap();
    let runtime = runtime().unwrap();
    let mut retained: Vec<Vec<CudaFloat32Storage>> = Vec::new();
    for mode in ["direct", "selected VM", "legacy VM"] {
        let kernel = if mode == "legacy VM" {
            jit::Kernel::compile(&graph, 0).unwrap()
        } else {
            let direct = mode == "direct";
            let identity = jit::ExecutableIdentity::new(
                &graph,
                &addresses,
                0,
                jit::Kernel::checked_context(0).unwrap(),
                direct.then_some(&program),
            );
            let compilation = if direct {
                crate::pointwise_ir::program::Compilation {
                    source: program.direct_source(&graph, &addresses).unwrap().unwrap(),
                    erf: false,
                }
            } else {
                graph.compilation(&addresses).unwrap()
            };
            jit::Kernel::compile_selected(&graph, 0, addresses.clone(), compilation, identity)
                .unwrap()
        };
        let plan = PointwisePlan::new(&kernel, 3, &program).unwrap();
        let empty_program = Program::build(&graph, &addresses, 0, &[0, 1], false).unwrap();
        let empty_plan = PointwisePlan::new(&kernel, 0, &empty_program).unwrap();
        for phase in [
            "success",
            "success",
            "allocation",
            "launch",
            "completion",
            "both",
            "empty",
        ] {
            let empty = phase == "empty";
            let scratch = mode != "direct";
            let _reset = ResetObservation;
            OBSERVATION.with(|slot| {
                *slot.borrow_mut() = Some(Observation {
                    phase,
                    scratch,
                    allocations: 0,
                    pointers: Vec::new(),
                    dropped: Vec::new(),
                    launches: 0,
                    completions: 0,
                    protected: vec![input.data_ptr, plan.instructions.as_ref().unwrap().data_ptr],
                    input: Arc::downgrade(&input),
                    kernel: Arc::downgrade(&kernel),
                });
            });
            let result = input.pointwise_jit(
                &input,
                [0, 0],
                if empty { 0 } else { 3 },
                if empty { [0, 0] } else { [3, 3] },
                &kernel,
                &[],
                if empty { &empty_plan } else { &plan },
            );
            let state = OBSERVATION.with(|slot| slot.borrow_mut().take().unwrap());
            assert_eq!(state.allocations, 2 + usize::from(scratch && !empty));
            let executed = usize::from(!empty && phase != "allocation");
            assert_eq!(state.launches, executed);
            assert_eq!(state.completions, executed);
            assert!(state.protected.iter().all(|p| !state.dropped.contains(p)));
            match phase {
                "success" => {
                    let outputs = result.unwrap();
                    assert_eq!(state.dropped.len(), usize::from(scratch));
                    assert_eq!(outputs[0].copy_range(0, 3).unwrap(), [-1., 2., -3.]);
                    assert_eq!(outputs[1].copy_range(0, 3).unwrap(), [1., -2., 3.]);
                    assert_ne!(outputs[0].data_ptr, outputs[1].data_ptr);
                    for output in &outputs {
                        assert_ne!(output.data_ptr, input.data_ptr);
                        assert!(
                            retained
                                .iter()
                                .flatten()
                                .all(|old| old.data_ptr != output.data_ptr)
                        );
                    }
                    retained.push(outputs);
                }
                "empty" => {
                    assert!(result.unwrap().iter().all(|output| output.elements == 0));
                    assert!(state.pointers.iter().all(|p| *p == 0));
                }
                _ => {
                    let expected = if phase == "both" { "launch" } else { phase };
                    assert!(matches!(result, Err(actual) if actual == injected(expected)));
                    assert_eq!(state.dropped.len(), state.pointers.len());
                    if executed != 0 {
                        assert!(!CACHE_HEALTHY.load(Ordering::Relaxed));
                    }
                }
            }
            let mut current = -1;
            runtime
                .check(
                    unsafe { (runtime.get_device)(&raw mut current) },
                    "cudaGetDevice",
                )
                .unwrap();
            assert_eq!(current, 0);
            assert_eq!(input.copy_range(0, 3).unwrap(), [1., -2., 3.]);
            eprintln!(
                "canonical {mode}/{phase}: allocations={}, launches={}, completions={}",
                state.allocations, state.launches, state.completions
            );
        }
    }
    for outputs in retained {
        assert_eq!(outputs[0].copy_range(0, 3).unwrap(), [-1., 2., -3.]);
    }
    eprintln!("GPU0 only: different-active-device restoration is not exercised");
}
