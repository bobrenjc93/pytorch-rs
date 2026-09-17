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
            assert_eq!(input.copy_range(0, 2).unwrap(), [2., 4.]);
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
        scalar_count: 0,
        column_certificate: 132,
        row_hint: rows.unwrap_or(128),
    }
}
fn gpu() -> bool {
    if crate::cuda::device_count() == 0 {
        eprintln!("MISSING COVERAGE: leading sum CUDA unavailable");
        return false;
    }
    let _guard = runtime().unwrap().guard(0).unwrap();
    current_context().unwrap();
    let driver = driver().unwrap();
    let mut ordinal = 0;
    let mut properties = [0; 5];
    // SAFETY: documented attributes and writable integers under the device guard.
    unsafe {
        driver
            .check((driver.device)(&raw mut ordinal), "cuCtxGetDevice")
            .unwrap();
        for (value, attribute) in properties.iter_mut().zip([75, 76, 10, 16, 39]) {
            driver
                .check(
                    (driver.attribute)(value, attribute, ordinal),
                    "cuDeviceGetAttribute(leading sum test prerequisite)",
                )
                .unwrap();
        }
    }
    // Positive fixtures reach C=256; query failures above are test failures.
    if let Err(error) = validate_device_properties(256, properties) {
        eprintln!("MISSING COVERAGE: leading sum device {properties:?}: {error}");
        return false;
    }
    true
}

#[test]
fn policy_identity_and_large_extents_are_checked_without_allocation() {
    let exact = descriptor(
        Some(129),
        false,
        Divisor::Dimension {
            axis: 0,
            certificate: Some(129),
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
    assert_eq!(exact.segments(), 2);
    assert_eq!(dynamic.segments(), 1);
    assert_ne!(exact.identity(0, 1), dynamic.identity(0, 1));
    assert!(exact.validate_shape(&[128, 132]).is_err());
    assert!(dynamic.validate_shape(&[193, 132]).is_ok());
    for rows in [0, 1, 3, 32, 64, 257, usize::MAX] {
        assert!(dynamic.validate_shape(&[rows, 132]).is_err());
    }
    for columns in [0, 1, 128, 131, 133, 255, 257, usize::MAX] {
        let mut bad = dynamic.clone();
        bad.column_certificate = columns as u64;
        assert!(bad.validate_shape(&[128, columns]).is_err());
    }
    for hint in [0, 64, 257, u64::MAX] {
        let mut bad = dynamic.clone();
        bad.row_hint = hint;
        assert!(bad.validate().is_err());
    }
    assert_eq!(blocks(usize::MAX), 65535);
    assert!(crate::pointwise_ir::indexing::Layout::new(&[usize::MAX, 2]).is_err());
    let mut bad = exact.clone();
    bad.divisor = Divisor::Dimension {
        axis: 0,
        certificate: None,
    };
    assert!(bad.validate().is_err());
    for bits in [(-0f64).to_bits(), 2f64.to_bits(), f64::NAN.to_bits()] {
        assert!(
            descriptor(
                Some(128),
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
                Some(128),
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
            Some(128),
            false,
            Divisor::Runtime {
                slot: 64,
                negative: false
            }
        )
        .validate()
        .is_err()
    );
    assert!(source(&dynamic).contains("__ull2float_rn(rows)"));
    assert!(source(&dynamic).contains("div.full.f32"));
    assert!(!source(&dynamic).contains("div.rn"));
    assert!(validate_device_properties(132, [9, 0, 32, 132, 2048]).is_ok());
    for properties in [
        [8, 0, 32, 132, 2048],
        [9, 1, 32, 132, 2048],
        [9, 0, 64, 132, 2048],
        [9, 0, 32, 0, 2048],
        [9, 0, 32, 132, 0],
        [9, 0, 32, 2, 2048],
        [9, 0, 32, 132, 8],
    ] {
        assert!(validate_device_properties(132, properties).is_err());
    }
}

#[test]
fn complete_scalar_abi_and_canonical_segments_define_identity_and_arithmetic() {
    let base = descriptor(None, false, Divisor::None);
    for count in [0, 1, 64] {
        let spec = LeadingSum {
            scalar_count: count,
            ..base.clone()
        };
        assert!(spec.validate().is_ok());
        assert_eq!(spec.scalar_count(), count);
    }
    assert!(
        LeadingSum {
            scalar_count: 65,
            ..base.clone()
        }
        .validate()
        .is_err()
    );
    for count in [1, 64] {
        for slot in [0, count - 1] {
            let spec = LeadingSum {
                scalar_count: count,
                divisor: Divisor::Runtime {
                    slot,
                    negative: true,
                },
                ..base.clone()
            };
            assert!(spec.validate().is_ok());
            assert!(source(&spec).contains("full_divide(value, scalar)"));
            assert!(
                LeadingSum {
                    divisor: Divisor::Runtime {
                        slot: count,
                        negative: false
                    },
                    ..spec
                }
                .validate()
                .is_err()
            );
        }
    }
    let same_segments = LeadingSum {
        row_hint: 65,
        ..base.clone()
    };
    assert_eq!(same_segments.identity(0, 1), base.identity(0, 1));
    for changed in [
        LeadingSum {
            row_hint: 129,
            ..base.clone()
        },
        LeadingSum {
            scalar_count: 1,
            ..base.clone()
        },
        LeadingSum {
            column_certificate: 256,
            ..base.clone()
        },
        LeadingSum {
            keepdim: true,
            ..base.clone()
        },
    ] {
        assert_ne!(changed.identity(0, 1), base.identity(0, 1));
    }
    assert_ne!(base.identity(0, 1), base.identity(0, 2));
    assert_ne!(base.identity(0, 1), base.identity(1, 1));
    for certificate in [None, Some(128), Some(256)] {
        assert!(
            LeadingSum {
                divisor: Divisor::Dimension {
                    axis: 1,
                    certificate
                },
                ..base.clone()
            }
            .validate()
            .is_err()
        );
    }
    assert!(
        LeadingSum {
            divisor: Divisor::Dimension {
                axis: 1,
                certificate: Some(132)
            },
            ..base.clone()
        }
        .validate()
        .is_ok()
    );
}

#[test]
fn seeded_segment_source_preserves_masks_halves_and_epilogue_order() {
    let code = source(&LeadingSum {
        row_hint: 193,
        ..descriptor(None, false, Divisor::None)
    });
    assert!(code.contains("segment < 2u"));
    assert!(code.contains("if (position < length)"));
    assert!(code.contains("row < rows && column < columns ? input[row * columns + column] : 0.0f"));
    let lo = code
        .find("__fadd_rn(__fadd_rn(p[0], p[2]), __fadd_rn(p[1], p[3]))")
        .unwrap();
    let hi = code
        .find("__fadd_rn(__fadd_rn(p[4], p[6]), __fadd_rn(p[5], p[7]))")
        .unwrap();
    let tree = code.find("step = 4").unwrap();
    let halves = code
        .find("float total = __fadd_rn(lo[0][threadIdx.x], hi[0][threadIdx.x])")
        .unwrap();
    let segments = code
        .find("value = segment == 0 ? total : __fadd_rn(value, total)")
        .unwrap();
    let output = code.find("output[column] = value").unwrap();
    assert!(lo < hi && hi < tree && tree < halves && halves < segments && segments < output);
}

#[test]
fn withdrawn_shapes_and_invalid_storage_ranges_reject_before_allocation() {
    if !gpu() {
        return;
    }
    let input = CudaFloat32Storage::from_host(&vec![0.; 128 * 132], 0).unwrap();
    let mut kernel = Kernel::compile(
        descriptor(None, false, Divisor::None),
        0,
        Module::checked_context(0).unwrap(),
    )
    .unwrap();
    let _reset = ResetAudit;
    AUDIT.with(|audit| {
        *audit.borrow_mut() = Some(Audit {
            failure: None,
            both: false,
            events: vec![],
            kernel: Arc::downgrade(&kernel),
        });
    });
    for shape in [
        [0, 132],
        [32, 1],
        [64, 132],
        [257, 132],
        [128, 0],
        [128, 255],
        [usize::MAX, 132],
    ] {
        assert!(input.leading_sum(0, shape, &kernel, &[]).is_err());
    }
    for offset in [1, usize::MAX] {
        assert!(matches!(
            input.leading_sum(offset, [128, 132], &kernel, &[]),
            Err(TensorError::IndexCalculationOverflow)
        ));
    }
    assert!(input.leading_sum(0, [128, 132], &kernel, &[1.]).is_err());
    assert!(AUDIT.with(|audit| audit.borrow_mut().take().unwrap().events.is_empty()));
    let context = kernel.context;
    Arc::get_mut(&mut kernel).unwrap().module.context = usize::MAX;
    assert!(input.leading_sum(0, [128, 132], &kernel, &[]).is_err());
    Arc::get_mut(&mut kernel).unwrap().module.context = context;
}

#[test]
fn seeded_segments_write_admitted_bounds_tails_and_current_offsets() {
    if !gpu() {
        return;
    }
    let cases: [(usize, usize, Vec<f32>, Vec<f32>); 4] = [
        (65, 132, vec![1.; 65 * 132], vec![65.; 132]),
        (128, 256, vec![-0.; 128 * 256], vec![0.; 256]),
        (193, 252, vec![1.; 193 * 252], vec![193.; 252]),
        (256, 132, vec![1.; 256 * 132], vec![256.; 132]),
    ];
    for (rows, cols, values, expected) in cases {
        let input = Tensor::from_vec(values, [rows, cols])
            .unwrap()
            .try_copy_cpu_to_cuda(Device::Cuda(0))
            .unwrap();
        for keepdim in [false, true] {
            let host = HostLeadingSum::new(
                &[&input],
                LeadingSum {
                    column_certificate: cols as u64,
                    ..descriptor(Some(rows as u64), keepdim, Divisor::None)
                },
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
    let mut values = vec![1.; 256 * 132];
    values[128 * 132..].fill(2.);
    let base = Tensor::from_vec(values, [256, 132])
        .unwrap()
        .try_copy_cpu_to_cuda(Device::Cuda(0))
        .unwrap();
    let mut input = base.slice_dimension(0, 0, 128).unwrap();
    let host = HostLeadingSum::new(&[&input], descriptor(Some(128), false, Divisor::None)).unwrap();
    let kernel = host.compile().unwrap();
    let weak = Arc::downgrade(&kernel);
    let prepared = host.bind(kernel).unwrap();
    let old = prepared.run(&[&input], &[]).unwrap().remove(0);
    input = base.slice_dimension(0, 128, 128).unwrap();
    let new = prepared.run(&[&input], &[]).unwrap().remove(0);
    assert_eq!(
        old.try_copy_cuda_to_cpu().unwrap().try_to_vec().unwrap(),
        vec![128.; 132]
    );
    assert_eq!(
        new.try_copy_cuda_to_cpu().unwrap().try_to_vec().unwrap(),
        vec![256.; 132]
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
    let mut values = vec![0.; 128 * 132];
    values[..132].fill(2f32.powi(-120));
    let input = Tensor::from_vec(values, [128, 132])
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
        let spec = LeadingSum {
            scalar_count: scalars.len(),
            ..descriptor(Some(128), false, divisor)
        };
        let host = HostLeadingSum::new(&[&input], spec).unwrap();
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
            assert_eq!(output, vec![f32::INFINITY; 132]);
        } else {
            assert_eq!(output, vec![1024.; 132]);
        }
    }
}

#[test]
fn actual_launch_and_completed_error_paths_preserve_owners_and_first_error() {
    if !gpu() {
        return;
    }
    let mut values = vec![0.; 128 * 132];
    values[0] = 2.;
    values[1] = 4.;
    values[132] = 4.;
    let input = Arc::new(CudaFloat32Storage::from_host(&values, 0).unwrap());
    let kernel = Kernel::compile(
        descriptor(Some(128), false, Divisor::None),
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
                events: vec![],
                kernel: Arc::downgrade(&kernel),
            });
        });
        let result = input.leading_sum(0, [128, 132], &kernel, &[]);
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
fn admitted_input_still_validates_context_and_exact_bound_identity() {
    if !gpu() {
        return;
    }
    let input = Tensor::from_vec(vec![1.; 128 * 132], [128, 132])
        .unwrap()
        .try_copy_cpu_to_cuda(Device::Cuda(0))
        .unwrap();
    let host = HostLeadingSum::new(&[&input], descriptor(Some(128), false, Divisor::None)).unwrap();
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
    assert_eq!(prepared.run(&[&input], &[]).unwrap()[0].shape(), [132]);
}

#[test]
fn empty_keepdim_layout_does_not_change_the_shared_pointwise_layout() {
    let layout = crate::pointwise_ir::indexing::Layout::new(&[1, 0]).unwrap();
    assert_eq!(layout.shape, [1, 0]);
    assert_eq!(layout.strides, [1, 1]);
    assert_eq!(layout.elements, 0);
}

fn check_prepared_outputs(keepdim: bool, divisor: Divisor) {
    let rows = 128;
    let input = Tensor::from_vec(vec![1.; rows * 132], [rows, 132])
        .unwrap()
        .try_copy_cpu_to_cuda(Device::Cuda(0))
        .unwrap();
    let expected_stride = if keepdim { vec![132, 1] } else { vec![1] };
    let expected = match divisor {
        Divisor::None => 128.,
        Divisor::Dimension { .. } => 1.,
        _ => 64.,
    };
    // Complete dense ABI includes unused slots even when there is no divisor.
    let spec = LeadingSum {
        scalar_count: 64,
        ..descriptor(None, keepdim, divisor)
    };
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
    assert_eq!(prepared.input_shapes(), &[vec![rows, 132]]);
    let before_bytes = prepared.retained_bytes();
    let first = prepared.run(&[&input], &scalars).unwrap().remove(0);
    let second = prepared.run(&[&input], &scalars).unwrap().remove(0);
    for output in [&first, &second] {
        assert_eq!(
            output.shape(),
            if keepdim { vec![1, 132] } else { vec![132] }
        );
        assert_eq!(output.stride(), expected_stride);
        assert_eq!(output.storage_offset(), 0);
        assert_eq!(output.dtype(), crate::DType::Float32);
        assert_eq!(output.device(), Device::Cuda(0));
        assert!(!output.requires_grad());
        assert_eq!(
            output.try_copy_cuda_to_cpu().unwrap().try_to_vec().unwrap(),
            vec![expected; 132]
        );
        assert!(!output.shares_storage_with(&input));
    }
    // Storage ownership is checked independently of the numerical result.
    assert!(!first.shares_storage_with(&second));
    assert_eq!(prepared.retained_bytes(), before_bytes);
    let wrong_shape = Tensor::from_vec(vec![0.0; (rows + 1) * 132], [rows + 1, 132])
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
    drop(kernel);
    assert_eq!(weak.strong_count(), 1);
    drop(prepared);
    assert!(weak.upgrade().is_none());
    assert_eq!(first.stride(), expected_stride);
    assert!(!first.shares_storage_with(&second));
}

#[test]
fn keepdim_host_bind_and_reuse_preserve_strides_owners_and_complete_arity() {
    if !gpu() {
        return;
    }
    for keepdim in [false, true] {
        for divisor in [
            Divisor::None,
            Divisor::Constant {
                kind: ScalarKind::Integer,
                bits: 2f64.to_bits(),
            },
            Divisor::Runtime {
                slot: 1,
                negative: false,
            },
            Divisor::Dimension {
                axis: 0,
                certificate: None,
            },
        ] {
            check_prepared_outputs(keepdim, divisor);
        }
    }
}
