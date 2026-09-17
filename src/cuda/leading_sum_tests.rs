//! Untimed tests of the real leading-sum storage and prepared owners.
use super::*;
use crate::{
    Device, Tensor,
    cuda::{CACHE_HEALTHY, CudaFloat32Storage, Ordering, runtime},
    tensor::leading_sum::{HostLeadingSum, ScalarKind},
};
use std::cell::RefCell;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Phase {
    Allocate,
    Launch,
    Complete,
}
struct Audit {
    failure: Option<Phase>,
    both: bool,
    empty: bool,
    events: Vec<Phase>,
    kernel: std::sync::Weak<Kernel>,
}
thread_local! { static AUDIT: RefCell<Option<Audit>> = const {RefCell::new(None)}; }
pub(crate) fn observe(
    phase: Phase,
    input: &CudaFloat32Storage,
    kernel: &Kernel,
    outputs: &[CudaFloat32Storage],
) -> Result<(), TensorError> {
    AUDIT.with(|audit| {
        let mut audit = audit.borrow_mut();
        if let Some(audit) = audit.as_mut() {
            assert!(audit.kernel.upgrade().is_some());
            assert_eq!(kernel.device, input.device_index);
            if audit.empty {
                assert_eq!(
                    phase,
                    Phase::Allocate,
                    "empty columns must not launch or complete"
                );
                assert_eq!(input.elements, 0);
                assert!(outputs.is_empty());
            } else {
                assert_eq!(input.copy_range(0, 2).unwrap(), [2., 4.]);
            }
            audit.events.push(phase);
            if phase != Phase::Allocate {
                assert_eq!(outputs.len(), 1);
                assert_ne!(outputs[0].data_ptr, input.data_ptr);
                if phase == Phase::Complete {
                    assert_eq!(outputs[0].copy_range(0, 1).unwrap(), [6.]);
                }
            }
            if audit.failure == Some(phase) || (audit.both && phase != Phase::Allocate) {
                return Err(invalid(format!("injected leading sum {phase:?}")));
            }
        }
        Ok(())
    })
}
struct ResetAudit;
impl Drop for ResetAudit {
    fn drop(&mut self) {
        AUDIT.with(|audit| *audit.borrow_mut() = None);
    }
}
fn descriptor(rows: Option<u64>, keepdim: bool, divisor: Divisor) -> LeadingSum {
    LeadingSum {
        axis: 0,
        keepdim,
        row_certificate: rows,
        divisor,
    }
}
fn gpu() -> bool {
    if crate::cuda::device_count() == 0 {
        eprintln!("MISSING COVERAGE: leading sum CUDA unavailable");
        false
    } else {
        true
    }
}

#[test]
fn policy_identity_and_large_extents_are_checked_without_allocation() {
    let static_three = descriptor(
        Some(3),
        false,
        Divisor::Dimension {
            axis: 0,
            certificate: Some(3),
        },
    );
    let dynamic = descriptor(
        None,
        false,
        Divisor::Dimension {
            axis: 0,
            certificate: None,
        },
    );
    assert_eq!(static_three.reduction(), Reduction::OrderedSmall(3));
    assert_eq!(dynamic.reduction(), Reduction::Tree8);
    assert_ne!(static_three.identity(0, 1), dynamic.identity(0, 1));
    assert!(static_three.validate_shape(&[5, 9]).is_err());
    assert!(dynamic.validate_shape(&[1, 9]).is_err());
    assert!(dynamic.validate_shape(&[3, 9]).is_ok());
    assert_eq!(blocks(usize::MAX), 65535);
    assert!(crate::pointwise_ir::indexing::Layout::new(&[usize::MAX, 2]).is_err());
    let mut bad = static_three.clone();
    bad.divisor = Divisor::Dimension {
        axis: 0,
        certificate: None,
    };
    assert!(bad.validate().is_err());
    for bits in [(-0f64).to_bits(), 2f64.to_bits(), f64::NAN.to_bits()] {
        assert!(
            descriptor(
                Some(2),
                false,
                Divisor::Constant {
                    kind: ScalarKind::Boolean,
                    bits
                }
            )
            .validate()
            .is_err()
        );
    }
    for bits in [0.5f64.to_bits(), f64::INFINITY.to_bits()] {
        assert!(
            descriptor(
                Some(2),
                false,
                Divisor::Constant {
                    kind: ScalarKind::Integer,
                    bits
                }
            )
            .validate()
            .is_err()
        );
    }
    assert!(
        descriptor(
            Some(2),
            false,
            Divisor::Runtime {
                slot: 4096,
                negative: false
            }
        )
        .validate()
        .is_err()
    );
    assert!(source(&dynamic).contains("__ull2float_rn(rows)"));
    assert!(source(&dynamic).contains("div.full.f32"));
    assert!(!source(&dynamic).contains("div.rn"));
}

#[test]
fn selected_native_policies_write_empty_singleton_ordered_tail_and_current_offsets() {
    if !gpu() {
        return;
    }
    let cases: [(usize, usize, Vec<f32>, Vec<f32>); 5] = [
        (0, 3, vec![], vec![0.; 3]),
        (1, 1, vec![-0.], vec![-0.]),
        (3, 1, vec![16_777_216., 1., -16_777_216.], vec![0.]),
        (9, 35, vec![1.; 315], vec![9.; 35]),
        (3, 0, vec![], vec![]),
    ];
    for (rows, cols, values, expected) in cases {
        let input = Tensor::from_vec(values, [rows, cols])
            .unwrap()
            .try_copy_cpu_to_cuda(Device::Cuda(0))
            .unwrap();
        for keepdim in [false, true] {
            let host = HostLeadingSum::new(
                &[&input],
                descriptor(Some(rows as u64), keepdim, Divisor::None),
            )
            .unwrap();
            let kernel = host.compile().unwrap();
            assert!(kernel.ptx.contains("torch_rs_leading_sum"));
            let prepared = host.bind(Arc::clone(&kernel)).unwrap();
            assert!(Arc::ptr_eq(prepared.kernel(), &kernel));
            let output = prepared.run(&[&input], &[]).unwrap().remove(0);
            let actual = output.try_copy_cuda_to_cpu().unwrap().try_to_vec().unwrap();
            assert_eq!(
                actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(
                output.shape(),
                if keepdim { vec![1, cols] } else { vec![cols] }
            );
            assert!(!output.shares_storage_with(&input));
            assert_eq!(prepared.input_shapes(), &[vec![rows, cols]]);
            assert!(
                prepared.retained_bytes()
                    > size_of::<crate::tensor::leading_sum::PreparedLeadingSum>()
            );
        }
    }
    let base = Tensor::from_vec(vec![1., 2., 3., 4.], [4, 1])
        .unwrap()
        .try_copy_cpu_to_cuda(Device::Cuda(0))
        .unwrap();
    let mut input = base.slice_dimension(0, 0, 2).unwrap();
    let host = HostLeadingSum::new(&[&input], descriptor(Some(2), false, Divisor::None)).unwrap();
    let kernel = host.compile().unwrap();
    let weak = Arc::downgrade(&kernel);
    let prepared = host.bind(kernel).unwrap();
    let old = prepared.run(&[&input], &[]).unwrap().remove(0);
    input = base.slice_dimension(0, 2, 2).unwrap();
    let new = prepared.run(&[&input], &[]).unwrap().remove(0);
    assert_eq!(
        old.try_copy_cuda_to_cpu().unwrap().try_to_vec().unwrap(),
        [3.]
    );
    assert_eq!(
        new.try_copy_cuda_to_cpu().unwrap().try_to_vec().unwrap(),
        [7.]
    );
    assert!(!old.shares_storage_with(&new));
    assert!(prepared.run(&[&base], &[]).is_err());
    assert!(prepared.run(&[&input], &[1.]).is_err());
    drop(prepared);
    assert!(weak.upgrade().is_none());
}

#[test]
fn constant_reciprocal_and_runtime_full_divide_compile_distinct_native_arithmetic() {
    if !gpu() {
        return;
    }
    let input = Tensor::from_vec(vec![2f32.powi(-120)], [1, 1])
        .unwrap()
        .try_copy_cpu_to_cuda(Device::Cuda(0))
        .unwrap();
    for (divisor, scalars, infinite) in [
        (
            Divisor::Constant {
                kind: ScalarKind::Float,
                bits: 2f64.powi(-130).to_bits(),
            },
            vec![],
            true,
        ),
        (
            Divisor::Runtime {
                slot: 0,
                negative: false,
            },
            vec![f32::from_bits(1 << 19)], // Exact 2^-130, independent of host powi intermediates.
            false,
        ),
    ] {
        let host = HostLeadingSum::new(&[&input], descriptor(Some(1), false, divisor)).unwrap();
        let kernel = host.compile().unwrap();
        if !infinite {
            assert!(kernel.ptx.contains("div.full.f32"));
        }
        let output = host
            .bind(kernel)
            .unwrap()
            .run(&[&input], &scalars)
            .unwrap()
            .remove(0)
            .try_copy_cuda_to_cpu()
            .unwrap()
            .try_to_vec()
            .unwrap();
        if infinite {
            assert_eq!(output, [f32::INFINITY]);
        } else {
            assert_eq!(output, [1024.]);
        }
    }
}

#[test]
fn actual_launch_and_completed_error_paths_preserve_owners_and_first_error() {
    if !gpu() {
        return;
    }
    let input = Arc::new(CudaFloat32Storage::from_host(&[2., 4.], 0).unwrap());
    let kernel = Kernel::compile(
        descriptor(Some(2), false, Divisor::None),
        0,
        Module::checked_context(0).unwrap(),
    )
    .unwrap();
    for (failure, both) in [
        (None, false),
        (Some(Phase::Allocate), false),
        (Some(Phase::Launch), false),
        (Some(Phase::Complete), false),
        (Some(Phase::Launch), true),
    ] {
        let _reset = ResetAudit;
        AUDIT.with(|a| {
            *a.borrow_mut() = Some(Audit {
                failure,
                both,
                empty: false,
                events: vec![],
                kernel: Arc::downgrade(&kernel),
            });
        });
        let result = input.leading_sum(0, [2, 1], &kernel, &[]);
        let audit = AUDIT.with(|a| a.borrow_mut().take().unwrap());
        if failure == Some(Phase::Allocate) {
            assert_eq!(audit.events, [Phase::Allocate]);
        } else {
            assert_eq!(
                audit.events,
                [Phase::Allocate, Phase::Launch, Phase::Complete]
            );
        }
        match failure {
            None => assert_eq!(result.unwrap().copy_range(0, 1).unwrap(), [6.]),
            Some(phase) => {
                assert!(
                    matches!(result,Err(error) if error.to_string().contains(&format!("{phase:?}")))
                );
            }
        }
        if matches!(failure, Some(Phase::Launch | Phase::Complete)) {
            assert!(!CACHE_HEALTHY.load(Ordering::Relaxed));
        }
        let rt = runtime().unwrap();
        let mut ordinal = -1;
        rt.check(
            unsafe { (rt.get_device)(&raw mut ordinal) },
            "cudaGetDevice",
        )
        .unwrap();
        assert_eq!(ordinal, 0);
    }
    eprintln!(
        "Leading sum failure injection follows real launch/synchronization; not an unfinished-device fault. GPU0 only; no different-active-device restoration claim."
    );
}

#[test]
fn empty_columns_still_validate_context_and_exact_bound_identity() {
    if !gpu() {
        return;
    }
    let input = Tensor::from_vec(vec![], [3, 0])
        .unwrap()
        .try_copy_cpu_to_cuda(Device::Cuda(0))
        .unwrap();
    let host = HostLeadingSum::new(&[&input], descriptor(Some(3), false, Divisor::None)).unwrap();
    let other = HostLeadingSum::new(&[&input], descriptor(None, false, Divisor::None)).unwrap();
    let mut kernel = host.compile().unwrap();
    assert!(other.bind(Arc::clone(&kernel)).is_err());
    let context = kernel.context;
    Arc::get_mut(&mut kernel).unwrap().module.context = usize::MAX;
    let result = host.bind(Arc::clone(&kernel));
    Arc::get_mut(&mut kernel).unwrap().module.context = context;
    assert!(matches!(result,Err(error) if error.to_string().contains("context mismatch")));
    let prepared = host.bind(kernel).unwrap();
    assert!(prepared.run(&[], &[]).is_err());
    assert!(prepared.run(&[&input, &input], &[]).is_err());
    assert_eq!(prepared.run(&[&input], &[]).unwrap()[0].shape(), [0]);
}

#[test]
fn empty_keepdim_layout_does_not_change_the_shared_pointwise_layout() {
    let layout = crate::pointwise_ir::indexing::Layout::new(&[1, 0]).unwrap();
    assert_eq!(layout.shape, [1, 0]);
    assert_eq!(layout.strides, [1, 1]);
    assert_eq!(layout.elements, 0);
}

fn check_empty_prepared_outputs(rows: usize, keepdim: bool, divisor: Divisor) {
    let input = Tensor::from_vec(vec![], [rows, 0])
        .unwrap()
        .try_copy_cpu_to_cuda(Device::Cuda(0))
        .unwrap();
    let spec = descriptor(Some(rows as u64), keepdim, divisor);
    let scalars = vec![2.0; spec.scalar_count()];
    let host = HostLeadingSum::new(&[&input], spec.clone()).unwrap();
    let mut kernel = host.compile().unwrap();
    let context = kernel.context;
    Arc::get_mut(&mut kernel).unwrap().module.context = usize::MAX;
    let rejected = host.bind(Arc::clone(&kernel));
    Arc::get_mut(&mut kernel).unwrap().module.context = context;
    assert!(matches!(rejected, Err(error) if error.to_string().contains("context mismatch")));
    let weak = Arc::downgrade(&kernel);
    let prepared = host.bind(Arc::clone(&kernel)).unwrap();
    assert!(Arc::ptr_eq(prepared.kernel(), &kernel));
    assert_eq!(prepared.input_shapes(), &[vec![rows, 0]]);
    let before_bytes = prepared.retained_bytes();
    let _reset = ResetAudit;
    AUDIT.with(|audit| {
        *audit.borrow_mut() = Some(Audit {
            failure: None,
            both: false,
            empty: true,
            events: vec![],
            kernel: Arc::downgrade(&kernel),
        });
    });
    let first = prepared.run(&[&input], &scalars).unwrap().remove(0);
    let second = prepared.run(&[&input], &scalars).unwrap().remove(0);
    for output in [&first, &second] {
        assert_eq!(output.shape(), if keepdim { vec![1, 0] } else { vec![0] });
        assert_eq!(output.stride(), if keepdim { vec![0, 1] } else { vec![1] });
        assert_eq!(output.storage_offset(), 0);
        assert_eq!(output.dtype(), crate::DType::Float32);
        assert_eq!(output.device(), Device::Cuda(0));
        assert!(!output.requires_grad());
        assert!(
            output
                .try_copy_cuda_to_cpu()
                .unwrap()
                .try_to_vec()
                .unwrap()
                .is_empty()
        );
        assert!(!output.shares_storage_with(&input));
    }
    // Empty allocations can share a null data address, but never their owners.
    assert!(!first.shares_storage_with(&second));
    assert_eq!(prepared.retained_bytes(), before_bytes);
    let wrong_shape = Tensor::from_vec(vec![0.0; rows], [rows, 1])
        .unwrap()
        .try_copy_cpu_to_cuda(Device::Cuda(0))
        .unwrap();
    assert!(matches!(prepared.run(&[&wrong_shape], &scalars), Err(error)
        if error.to_string().contains("input shape guard mismatch")));
    let mut wrong_scalars = scalars.clone();
    wrong_scalars.push(1.0);
    assert!(matches!(prepared.run(&[&input], &wrong_scalars), Err(error)
        if error.to_string().contains("runtime scalar arity mismatch")));
    let mut wrong_descriptor = spec.clone();
    wrong_descriptor.row_certificate = Some(rows as u64 + 1);
    assert!(HostLeadingSum::new(&[&input], wrong_descriptor).is_err());
    let opposite_keepdim = HostLeadingSum::new(
        &[&input],
        LeadingSum {
            keepdim: !keepdim,
            ..spec
        },
    )
    .unwrap();
    assert!(
        matches!(opposite_keepdim.bind(Arc::clone(&kernel)), Err(error)
        if error.to_string().contains("executable identity mismatch"))
    );
    let audit = AUDIT.with(|audit| audit.borrow_mut().take().unwrap());
    assert_eq!(audit.events, [Phase::Allocate, Phase::Allocate]);
    drop(kernel);
    assert_eq!(weak.strong_count(), 1);
    drop(prepared);
    assert!(weak.upgrade().is_none());
    assert_eq!(first.stride(), if keepdim { vec![0, 1] } else { vec![1] });
    assert!(!first.shares_storage_with(&second));
}

#[test]
fn empty_keepdim_host_bind_and_reuse_preserve_strides_owners_and_no_launch() {
    if !gpu() {
        return;
    }
    for rows in [0, 1, 5] {
        for keepdim in [false, true] {
            for divisor in [
                Divisor::None,
                Divisor::Runtime {
                    slot: 0,
                    negative: false,
                },
                Divisor::Dimension {
                    axis: 0,
                    certificate: Some(rows as u64),
                },
            ] {
                check_empty_prepared_outputs(rows, keepdim, divisor);
            }
        }
    }
}
