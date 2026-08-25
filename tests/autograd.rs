use pytorch_rs::{MemoryFormat, Tensor, TensorError, is_grad_enabled, no_grad};
use std::{
    sync::{Arc, Barrier},
    thread,
};

fn values(tensor: &Tensor) -> Vec<f32> {
    tensor.try_to_vec().unwrap()
}

const SIGMOID_AUTOGRAD_INPUT_BITS: [u32; 8] = [
    0x0000_0000,
    0x8000_0000,
    0x3f00_0000,
    0xbf00_0000,
    0x4185_1591,
    0x4185_1592,
    0xc2b1_7217,
    0xc2b1_7218,
];
const SIGMOID_AUTOGRAD_OUTPUT_BITS: [u32; 8] = [
    0x3f00_0000,
    0x3f00_0000,
    0x3f1f_597f,
    0x3ec1_4d03,
    0x3f7f_fffe,
    0x3f80_0000,
    0x0020_0010,
    0x0000_0000,
];
const SIGMOID_AUTOGRAD_GRADIENT_BITS: [u32; 8] = [
    0x3e80_0000,
    0xbf00_0000,
    0x3df0_a4d0,
    0xbd70_a4d0,
    0x34bf_fffe,
    0x8000_0000,
    0x00a0_0050,
    0x8000_0000,
];
const SIGMOID_AUTOGRAD_ACCUMULATED_GRADIENT_BITS: [u32; 8] = [
    0x3f00_0000,
    0xbf80_0000,
    0x3e70_a4d0,
    0xbdf0_a4d0,
    0x353f_fffe,
    0x8000_0000,
    0x0120_0050,
    0x8000_0000,
];
const TANH_AUTOGRAD_INPUT_BITS: [u32; 8] = [
    0x0000_0000,
    0x8000_0000,
    0x3f00_0000,
    0xbf00_0000,
    0x4110_2c66,
    0xc110_2c66,
    0x4110_2c67,
    0xc110_2c67,
];
const TANH_AUTOGRAD_OUTPUT_BITS: [u32; 8] = [
    0x0000_0000,
    0x8000_0000,
    0x3eec_9a9f,
    0xbeec_9a9f,
    0x3f7f_ffff,
    0xbf7f_ffff,
    0x3f80_0000,
    0xbf80_0000,
];
const TANH_AUTOGRAD_GRADIENT_BITS: [u32; 8] = [
    0x3f80_0000,
    0xc000_0000,
    0x3ec9_54a3,
    0xbe49_54a3,
    0x34c0_0000,
    0xb500_0000,
    0x0000_0000,
    0x8000_0000,
];
const TANH_AUTOGRAD_ACCUMULATED_GRADIENT_BITS: [u32; 8] = [
    0x4000_0000,
    0xc080_0000,
    0x3f49_54a3,
    0xbec9_54a3,
    0x3540_0000,
    0xb580_0000,
    0x0000_0000,
    0x8000_0000,
];

#[test]
fn square_sum_records_shared_leaf_once_and_accumulates_gradients() {
    let x = Tensor::from_vec(vec![-2.0, 0.5, 3.0], [3])
        .unwrap()
        .with_requires_grad(true);

    x.mul(&x).unwrap().sum().backward().unwrap();
    assert_eq!(values(&x.grad().unwrap().unwrap()), [-4.0, 1.0, 6.0]);

    x.mul(&x).unwrap().sum().backward().unwrap();
    assert_eq!(values(&x.grad().unwrap().unwrap()), [-8.0, 2.0, 12.0]);
}

#[test]
fn absolute_value_rejects_recording_before_planning_and_honors_no_grad() {
    let leaf = Tensor::from_vec(vec![-2.0, -0.0, 1.0, 4.0], [2, 2])
        .unwrap()
        .with_requires_grad(true);
    assert_eq!(
        leaf.abs(),
        Err(TensorError::AutogradRecordingUnsupported { operation: "abs" })
    );

    let extreme = Tensor::zeros([0])
        .unwrap()
        .reshape([0, i64::MAX, 3])
        .unwrap()
        .with_requires_grad(true);
    assert_eq!(
        extreme.abs(),
        Err(TensorError::AutogradRecordingUnsupported { operation: "abs" })
    );

    {
        let _guard = no_grad();
        let output = leaf.transpose(0, 1).unwrap().abs().unwrap();
        assert_eq!(output.shape(), [2, 2]);
        assert_eq!(output.stride(), [1, 2]);
        assert_eq!(output.storage_offset(), 0);
        assert!(!output.requires_grad());
        assert!(!output.shares_storage_with(&leaf));
        assert_eq!(
            output
                .logical_values()
                .map(f32::to_bits)
                .collect::<Vec<_>>(),
            [
                2.0_f32.to_bits(),
                1.0_f32.to_bits(),
                0.0_f32.to_bits(),
                4.0_f32.to_bits()
            ]
        );
        assert_eq!(extreme.abs(), Err(TensorError::StrideCalculationOverflow));
    }

    let detached = leaf.detach().unwrap().abs().unwrap();
    assert!(!detached.requires_grad());
}

#[test]
fn reciprocal_rejects_recording_before_planning_and_honors_no_grad() {
    let leaf = Tensor::from_vec(vec![-2.0, -0.0, 1.0, 4.0], [2, 2])
        .unwrap()
        .with_requires_grad(true);
    assert_eq!(
        leaf.reciprocal(),
        Err(TensorError::AutogradRecordingUnsupported {
            operation: "reciprocal",
        })
    );

    let extreme = Tensor::zeros([0])
        .unwrap()
        .reshape([0, i64::MAX, 3])
        .unwrap()
        .with_requires_grad(true);
    assert_eq!(
        extreme.reciprocal(),
        Err(TensorError::AutogradRecordingUnsupported {
            operation: "reciprocal",
        })
    );

    {
        let _guard = no_grad();
        let output = leaf.transpose(0, 1).unwrap().reciprocal().unwrap();
        assert_eq!(output.shape(), [2, 2]);
        assert_eq!(output.stride(), [1, 2]);
        assert_eq!(output.storage_offset(), 0);
        assert!(!output.requires_grad());
        assert!(!output.shares_storage_with(&leaf));
        assert_eq!(
            output
                .logical_values()
                .map(f32::to_bits)
                .collect::<Vec<_>>(),
            [
                (-0.5_f32).to_bits(),
                1.0_f32.to_bits(),
                f32::NEG_INFINITY.to_bits(),
                0.25_f32.to_bits()
            ]
        );
        assert_eq!(
            extreme.reciprocal(),
            Err(TensorError::StrideCalculationOverflow)
        );
    }

    let detached = leaf.detach().unwrap().reciprocal().unwrap();
    assert!(!detached.requires_grad());
}

#[test]
fn floor_records_reusable_zero_vjp_for_views_and_honors_no_grad() {
    let leaf = Tensor::from_vec(vec![-1.25, -0.0, 1.75, 4.5], [2, 2])
        .unwrap()
        .with_requires_grad(true);
    let output = leaf.transpose(0, 1).unwrap().floor().unwrap();
    assert_eq!(output.shape(), [2, 2]);
    assert_eq!(output.stride(), [1, 2]);
    assert_eq!(output.storage_offset(), 0);
    assert!(output.requires_grad());
    assert!(!output.is_leaf());
    assert!(!output.shares_storage_with(&leaf));
    assert_eq!(
        output
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        [
            (-2.0_f32).to_bits(),
            1.0_f32.to_bits(),
            (-0.0_f32).to_bits(),
            4.0_f32.to_bits(),
        ]
    );

    let loss = output.sum();
    loss.backward().unwrap();
    loss.backward().unwrap();
    assert!(
        leaf.grad()
            .unwrap()
            .unwrap()
            .logical_values()
            .all(|gradient| gradient.to_bits() == 0.0_f32.to_bits())
    );

    let extreme = Tensor::zeros([0])
        .unwrap()
        .reshape([0, i64::MAX, 3])
        .unwrap()
        .with_requires_grad(true);
    assert_eq!(extreme.floor(), Err(TensorError::StrideCalculationOverflow));

    {
        let _guard = no_grad();
        let output = leaf.transpose(0, 1).unwrap().floor().unwrap();
        assert_eq!(output.shape(), [2, 2]);
        assert_eq!(output.stride(), [1, 2]);
        assert_eq!(output.storage_offset(), 0);
        assert!(!output.requires_grad());
        assert!(output.is_leaf());
        assert!(!output.shares_storage_with(&leaf));
        assert_eq!(
            output
                .logical_values()
                .map(f32::to_bits)
                .collect::<Vec<_>>(),
            [
                (-2.0_f32).to_bits(),
                1.0_f32.to_bits(),
                (-0.0_f32).to_bits(),
                4.0_f32.to_bits(),
            ]
        );
        assert_eq!(extreme.floor(), Err(TensorError::StrideCalculationOverflow));
    }

    let detached = leaf.detach().unwrap().floor().unwrap();
    assert!(!detached.requires_grad());
    assert!(detached.is_leaf());
    assert!(!detached.shares_storage_with(&leaf));
}

#[test]
fn floor_zero_vjp_ignores_upstream_values_composes_and_accumulates() {
    let special =
        Tensor::from_vec(vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0], [4]).unwrap();
    let leaf = Tensor::from_vec(vec![-1.25, -0.0, 1.75, 4.5], [4])
        .unwrap()
        .with_requires_grad(true);
    leaf.floor()
        .unwrap()
        .mul(&special)
        .unwrap()
        .sum()
        .backward()
        .unwrap();
    assert!(
        leaf.grad()
            .unwrap()
            .unwrap()
            .logical_values()
            .all(|gradient| gradient.to_bits() == 0.0_f32.to_bits())
    );

    let accumulated = Tensor::from_vec(vec![-2.0, 0.0, 3.0], [3])
        .unwrap()
        .with_requires_grad(true);
    accumulated
        .mul_scalar(3.0)
        .unwrap()
        .sum()
        .backward()
        .unwrap();
    accumulated.floor().unwrap().sum().backward().unwrap();
    assert_eq!(
        values(&accumulated.grad().unwrap().unwrap()),
        [3.0, 3.0, 3.0]
    );

    let composed = Tensor::from_vec(vec![-0.5, 0.5], [2])
        .unwrap()
        .with_requires_grad(true);
    let composed_loss = composed.sin().unwrap().floor().unwrap().sum();
    composed_loss.backward().unwrap();
    assert_eq!(values(&composed.grad().unwrap().unwrap()), [0.0, 0.0]);
    assert_eq!(
        composed_loss.backward(),
        Err(TensorError::BackwardGraphFreed)
    );

    let empty = Tensor::zeros([2, 0, 3]).unwrap().with_requires_grad(true);
    empty
        .transpose(0, 2)
        .unwrap()
        .index([1])
        .unwrap()
        .floor()
        .unwrap()
        .sum()
        .backward()
        .unwrap();
    let empty_gradient = empty.grad().unwrap().unwrap();
    assert_eq!(empty_gradient.shape(), [2, 0, 3]);
    assert!(values(&empty_gradient).is_empty());
}

#[test]
fn ceil_records_reusable_zero_vjp_for_views_and_honors_no_grad() {
    let leaf = Tensor::from_vec(vec![-1.25, -0.0, 1.75, 4.5], [2, 2])
        .unwrap()
        .with_requires_grad(true);
    let output = leaf.transpose(0, 1).unwrap().ceil().unwrap();
    assert_eq!(output.shape(), [2, 2]);
    assert_eq!(output.stride(), [1, 2]);
    assert_eq!(output.storage_offset(), 0);
    assert!(output.requires_grad());
    assert!(!output.is_leaf());
    assert!(!output.shares_storage_with(&leaf));
    assert_eq!(
        output
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        [
            (-1.0_f32).to_bits(),
            2.0_f32.to_bits(),
            (-0.0_f32).to_bits(),
            5.0_f32.to_bits(),
        ]
    );

    let loss = output.sum();
    loss.backward().unwrap();
    loss.backward().unwrap();
    assert!(
        leaf.grad()
            .unwrap()
            .unwrap()
            .logical_values()
            .all(|gradient| gradient.to_bits() == 0.0_f32.to_bits())
    );

    let extreme = Tensor::zeros([0])
        .unwrap()
        .reshape([0, i64::MAX, 3])
        .unwrap()
        .with_requires_grad(true);
    assert_eq!(extreme.ceil(), Err(TensorError::StrideCalculationOverflow));

    {
        let _guard = no_grad();
        let output = leaf.transpose(0, 1).unwrap().ceil().unwrap();
        assert_eq!(output.shape(), [2, 2]);
        assert_eq!(output.stride(), [1, 2]);
        assert_eq!(output.storage_offset(), 0);
        assert!(!output.requires_grad());
        assert!(output.is_leaf());
        assert!(!output.shares_storage_with(&leaf));
        assert_eq!(
            output
                .logical_values()
                .map(f32::to_bits)
                .collect::<Vec<_>>(),
            [
                (-1.0_f32).to_bits(),
                2.0_f32.to_bits(),
                (-0.0_f32).to_bits(),
                5.0_f32.to_bits(),
            ]
        );
        assert_eq!(extreme.ceil(), Err(TensorError::StrideCalculationOverflow));
    }

    let detached = leaf.detach().unwrap().ceil().unwrap();
    assert!(!detached.requires_grad());
    assert!(detached.is_leaf());
    assert!(!detached.shares_storage_with(&leaf));
}

#[test]
fn ceil_zero_vjp_ignores_upstream_values_composes_and_accumulates() {
    let special =
        Tensor::from_vec(vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0], [4]).unwrap();
    let leaf = Tensor::from_vec(vec![-1.25, -0.0, 1.75, 4.5], [4])
        .unwrap()
        .with_requires_grad(true);
    leaf.ceil()
        .unwrap()
        .mul(&special)
        .unwrap()
        .sum()
        .backward()
        .unwrap();
    assert!(
        leaf.grad()
            .unwrap()
            .unwrap()
            .logical_values()
            .all(|gradient| gradient.to_bits() == 0.0_f32.to_bits())
    );

    let accumulated = Tensor::from_vec(vec![-2.0, 0.0, 3.0], [3])
        .unwrap()
        .with_requires_grad(true);
    accumulated
        .mul_scalar(3.0)
        .unwrap()
        .sum()
        .backward()
        .unwrap();
    accumulated.ceil().unwrap().sum().backward().unwrap();
    assert_eq!(
        values(&accumulated.grad().unwrap().unwrap()),
        [3.0, 3.0, 3.0]
    );

    let composed = Tensor::from_vec(vec![-0.5, 0.5], [2])
        .unwrap()
        .with_requires_grad(true);
    let composed_loss = composed.sin().unwrap().ceil().unwrap().sum();
    composed_loss.backward().unwrap();
    assert_eq!(values(&composed.grad().unwrap().unwrap()), [0.0, 0.0]);
    assert_eq!(
        composed_loss.backward(),
        Err(TensorError::BackwardGraphFreed)
    );

    let empty = Tensor::zeros([2, 0, 3]).unwrap().with_requires_grad(true);
    empty
        .transpose(0, 2)
        .unwrap()
        .index([1])
        .unwrap()
        .ceil()
        .unwrap()
        .sum()
        .backward()
        .unwrap();
    let empty_gradient = empty.grad().unwrap().unwrap();
    assert_eq!(empty_gradient.shape(), [2, 0, 3]);
    assert!(values(&empty_gradient).is_empty());
}

#[test]
fn trunc_records_reusable_zero_vjp_for_views_and_honors_no_grad() {
    let leaf = Tensor::from_vec(vec![-1.25, -0.0, 1.75, 4.5], [2, 2])
        .unwrap()
        .with_requires_grad(true);
    let output = leaf.transpose(0, 1).unwrap().trunc().unwrap();
    assert_eq!(output.shape(), [2, 2]);
    assert_eq!(output.stride(), [1, 2]);
    assert_eq!(output.storage_offset(), 0);
    assert!(output.requires_grad());
    assert!(!output.is_leaf());
    assert!(!output.shares_storage_with(&leaf));
    assert_eq!(
        output
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        [
            (-1.0_f32).to_bits(),
            1.0_f32.to_bits(),
            (-0.0_f32).to_bits(),
            4.0_f32.to_bits(),
        ]
    );

    let loss = output.sum();
    loss.backward().unwrap();
    loss.backward().unwrap();
    assert!(
        leaf.grad()
            .unwrap()
            .unwrap()
            .logical_values()
            .all(|gradient| gradient.to_bits() == 0.0_f32.to_bits())
    );

    let extreme = Tensor::zeros([0])
        .unwrap()
        .reshape([0, i64::MAX, 3])
        .unwrap()
        .with_requires_grad(true);
    assert_eq!(extreme.trunc(), Err(TensorError::StrideCalculationOverflow));

    {
        let _guard = no_grad();
        let output = leaf.transpose(0, 1).unwrap().trunc().unwrap();
        assert_eq!(output.shape(), [2, 2]);
        assert_eq!(output.stride(), [1, 2]);
        assert_eq!(output.storage_offset(), 0);
        assert!(!output.requires_grad());
        assert!(output.is_leaf());
        assert!(!output.shares_storage_with(&leaf));
        assert_eq!(
            output
                .logical_values()
                .map(f32::to_bits)
                .collect::<Vec<_>>(),
            [
                (-1.0_f32).to_bits(),
                1.0_f32.to_bits(),
                (-0.0_f32).to_bits(),
                4.0_f32.to_bits(),
            ]
        );
        assert_eq!(extreme.trunc(), Err(TensorError::StrideCalculationOverflow));
    }

    let detached = leaf.detach().unwrap().trunc().unwrap();
    assert!(!detached.requires_grad());
    assert!(detached.is_leaf());
    assert!(!detached.shares_storage_with(&leaf));
}

#[test]
fn trunc_zero_vjp_ignores_upstream_values_composes_and_accumulates() {
    let special =
        Tensor::from_vec(vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0], [4]).unwrap();
    let leaf = Tensor::from_vec(vec![-1.25, -0.0, 1.75, 4.5], [4])
        .unwrap()
        .with_requires_grad(true);
    leaf.trunc()
        .unwrap()
        .mul(&special)
        .unwrap()
        .sum()
        .backward()
        .unwrap();
    assert!(
        leaf.grad()
            .unwrap()
            .unwrap()
            .logical_values()
            .all(|gradient| gradient.to_bits() == 0.0_f32.to_bits())
    );

    let accumulated = Tensor::from_vec(vec![-2.0, 0.0, 3.0], [3])
        .unwrap()
        .with_requires_grad(true);
    accumulated
        .mul_scalar(3.0)
        .unwrap()
        .sum()
        .backward()
        .unwrap();
    accumulated.trunc().unwrap().sum().backward().unwrap();
    assert_eq!(
        values(&accumulated.grad().unwrap().unwrap()),
        [3.0, 3.0, 3.0]
    );

    let composed = Tensor::from_vec(vec![-0.5, 0.5], [2])
        .unwrap()
        .with_requires_grad(true);
    let composed_loss = composed.sin().unwrap().trunc().unwrap().sum();
    composed_loss.backward().unwrap();
    assert_eq!(values(&composed.grad().unwrap().unwrap()), [0.0, 0.0]);
    assert_eq!(
        composed_loss.backward(),
        Err(TensorError::BackwardGraphFreed)
    );

    let empty = Tensor::zeros([2, 0, 3]).unwrap().with_requires_grad(true);
    empty
        .transpose(0, 2)
        .unwrap()
        .index([1])
        .unwrap()
        .trunc()
        .unwrap()
        .sum()
        .backward()
        .unwrap();
    let empty_gradient = empty.grad().unwrap().unwrap();
    assert_eq!(empty_gradient.shape(), [2, 0, 3]);
    assert!(values(&empty_gradient).is_empty());
}

#[test]
fn sigmoid_differentiates_finite_owned_scalars_at_signed_zero_and_saturation() {
    // input, forward result, unit-upstream gradient
    const CASES: [(u32, u32, u32); 10] = [
        (0x0000_0000, 0x3f00_0000, 0x3e80_0000),
        (0x8000_0000, 0x3f00_0000, 0x3e80_0000),
        (0x0000_0001, 0x3f00_0000, 0x3e80_0000),
        (0x8000_0001, 0x3f00_0000, 0x3e80_0000),
        (0x3f00_0000, 0x3f1f_597f, 0x3e70_a4d0),
        (0xbf00_0000, 0x3ec1_4d03, 0x3e70_a4d0),
        (0x4185_1591, 0x3f7f_fffe, 0x33ff_fffe),
        (0x4185_1592, 0x3f80_0000, 0x0000_0000),
        (0xc2b1_7217, 0x0020_0010, 0x0020_0010),
        (0xc2b1_7218, 0x0000_0000, 0x0000_0000),
    ];

    for (input_bits, output_bits, gradient_bits) in CASES {
        let leaf = Tensor::from_vec(vec![f32::from_bits(input_bits)], [])
            .unwrap()
            .with_requires_grad(true);
        let output = leaf.sigmoid().unwrap();

        assert!(output.requires_grad());
        assert!(!output.is_leaf());
        assert!(output.shape().is_empty());
        assert!(output.stride().is_empty());
        assert_eq!(output.storage_offset(), 0);
        assert!(!output.shares_storage_with(&leaf));
        assert_eq!(output.item().unwrap().to_bits(), output_bits);

        output.backward().unwrap();
        assert_eq!(
            leaf.grad().unwrap().unwrap().item().unwrap().to_bits(),
            gradient_bits
        );
        assert_eq!(output.backward(), Err(TensorError::BackwardGraphFreed));
    }
}

#[test]
fn sigmoid_differentiates_owned_vectors_with_weighting_accumulation_and_graph_freeing() {
    let input = SIGMOID_AUTOGRAD_INPUT_BITS.map(f32::from_bits).to_vec();
    let weights = Tensor::from_vec(vec![1.0, -2.0, 0.5, -0.25, 3.0, -4.0, 5.0, -6.0], [8]).unwrap();
    let leaf = Tensor::from_vec(input.clone(), [8])
        .unwrap()
        .with_requires_grad(true);
    let output = leaf.sigmoid().unwrap();

    assert!(output.requires_grad());
    assert!(!output.is_leaf());
    assert_eq!(output.shape(), [8]);
    assert_eq!(output.stride(), [1]);
    assert_eq!(output.storage_offset(), 0);
    assert!(!output.shares_storage_with(&leaf));
    assert_eq!(
        output
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        SIGMOID_AUTOGRAD_OUTPUT_BITS
    );

    let loss = output.mul(&weights).unwrap().sum();
    loss.backward().unwrap();
    let gradient_bits = leaf
        .grad()
        .unwrap()
        .unwrap()
        .logical_values()
        .map(f32::to_bits)
        .collect::<Vec<_>>();
    assert_eq!(gradient_bits, SIGMOID_AUTOGRAD_GRADIENT_BITS);
    assert_eq!(loss.backward(), Err(TensorError::BackwardGraphFreed));
    assert_eq!(
        leaf.grad()
            .unwrap()
            .unwrap()
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        SIGMOID_AUTOGRAD_GRADIENT_BITS
    );

    let accumulated = Tensor::from_vec(input, [8])
        .unwrap()
        .with_requires_grad(true);
    for _ in 0..2 {
        accumulated
            .sigmoid()
            .unwrap()
            .mul(&weights)
            .unwrap()
            .sum()
            .backward()
            .unwrap();
    }
    assert_eq!(
        accumulated
            .grad()
            .unwrap()
            .unwrap()
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        SIGMOID_AUTOGRAD_ACCUMULATED_GRADIENT_BITS
    );

    let empty = Tensor::zeros([0]).unwrap().with_requires_grad(true);
    let empty_output = empty.sigmoid().unwrap();
    assert!(empty_output.requires_grad());
    assert!(!empty_output.is_leaf());
    assert_eq!(empty_output.shape(), [0]);
    assert_eq!(empty_output.stride(), [1]);
    assert!(!empty_output.shares_storage_with(&empty));
    let empty_loss = empty_output.sum();
    empty_loss.backward().unwrap();
    let empty_gradient = empty.grad().unwrap().unwrap();
    assert_eq!(empty_gradient.shape(), [0]);
    assert_eq!(empty_gradient.stride(), [1]);
    assert!(values(&empty_gradient).is_empty());
    assert_eq!(empty_loss.backward(), Err(TensorError::BackwardGraphFreed));
}

#[test]
fn sigmoid_differentiates_owned_matrices_with_weighting_accumulation_and_graph_freeing() {
    let input = SIGMOID_AUTOGRAD_INPUT_BITS.map(f32::from_bits).to_vec();
    let weights =
        Tensor::from_vec(vec![1.0, -2.0, 0.5, -0.25, 3.0, -4.0, 5.0, -6.0], [2, 4]).unwrap();
    let leaf = Tensor::from_vec(input.clone(), [2, 4])
        .unwrap()
        .with_requires_grad(true);
    let output = leaf.sigmoid().unwrap();

    assert!(output.requires_grad());
    assert!(!output.is_leaf());
    assert_eq!(output.shape(), [2, 4]);
    assert_eq!(output.stride(), [4, 1]);
    assert_eq!(output.storage_offset(), 0);
    assert_eq!(output.dtype(), leaf.dtype());
    assert_eq!(output.device(), leaf.device());
    assert!(!output.shares_storage_with(&leaf));
    assert_eq!(
        output
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        SIGMOID_AUTOGRAD_OUTPUT_BITS
    );

    let loss = output.mul(&weights).unwrap().sum();
    loss.backward().unwrap();
    let gradient = leaf.grad().unwrap().unwrap();
    assert_eq!(gradient.shape(), [2, 4]);
    assert_eq!(gradient.stride(), [4, 1]);
    assert_eq!(gradient.storage_offset(), 0);
    assert_eq!(gradient.dtype(), leaf.dtype());
    assert_eq!(gradient.device(), leaf.device());
    assert_eq!(
        gradient
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        SIGMOID_AUTOGRAD_GRADIENT_BITS
    );
    assert_eq!(loss.backward(), Err(TensorError::BackwardGraphFreed));
    assert_eq!(
        leaf.grad()
            .unwrap()
            .unwrap()
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        SIGMOID_AUTOGRAD_GRADIENT_BITS
    );

    let accumulated = Tensor::from_vec(input, [2, 4])
        .unwrap()
        .with_requires_grad(true);
    for _ in 0..2 {
        accumulated
            .sigmoid()
            .unwrap()
            .mul(&weights)
            .unwrap()
            .sum()
            .backward()
            .unwrap();
    }
    let accumulated_gradient = accumulated.grad().unwrap().unwrap();
    assert_eq!(accumulated_gradient.shape(), [2, 4]);
    assert_eq!(accumulated_gradient.stride(), [4, 1]);
    assert_eq!(
        accumulated_gradient
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        SIGMOID_AUTOGRAD_ACCUMULATED_GRADIENT_BITS
    );

    for (shape, expected_stride) in [([0, 0], [1, 1]), ([0, 3], [3, 1]), ([2, 0], [1, 1])] {
        let empty = Tensor::zeros(shape).unwrap().with_requires_grad(true);
        let empty_output = empty.sigmoid().unwrap();
        assert!(empty_output.requires_grad());
        assert!(!empty_output.is_leaf());
        assert_eq!(empty_output.shape(), shape);
        assert_eq!(empty_output.stride(), expected_stride);
        assert_eq!(empty_output.storage_offset(), 0);
        assert_eq!(empty_output.dtype(), empty.dtype());
        assert_eq!(empty_output.device(), empty.device());
        assert!(!empty_output.shares_storage_with(&empty));

        let empty_loss = empty_output.sum();
        empty_loss.backward().unwrap();
        let empty_gradient = empty.grad().unwrap().unwrap();
        assert_eq!(empty_gradient.shape(), shape);
        assert_eq!(empty_gradient.stride(), expected_stride);
        assert_eq!(empty_gradient.storage_offset(), 0);
        assert!(values(&empty_gradient).is_empty());
        assert_eq!(empty_loss.backward(), Err(TensorError::BackwardGraphFreed));
    }
}

#[test]
fn sigmoid_differentiates_owned_rank_three_tensors_with_singletons_and_empty_dimensions() {
    let input = SIGMOID_AUTOGRAD_INPUT_BITS.map(f32::from_bits).to_vec();
    let weights =
        Tensor::from_vec(vec![1.0, -2.0, 0.5, -0.25, 3.0, -4.0, 5.0, -6.0], [2, 1, 4]).unwrap();
    let leaf = Tensor::from_vec(input.clone(), [2, 1, 4])
        .unwrap()
        .with_requires_grad(true);
    let output = leaf.sigmoid().unwrap();

    assert!(output.requires_grad());
    assert!(!output.is_leaf());
    assert_eq!(output.shape(), [2, 1, 4]);
    assert_eq!(output.stride(), [4, 4, 1]);
    assert_eq!(output.storage_offset(), 0);
    assert_eq!(output.dtype(), leaf.dtype());
    assert_eq!(output.device(), leaf.device());
    assert!(!output.shares_storage_with(&leaf));
    assert_eq!(
        output
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        SIGMOID_AUTOGRAD_OUTPUT_BITS
    );

    let loss = output.mul(&weights).unwrap().sum();
    loss.backward().unwrap();
    let gradient = leaf.grad().unwrap().unwrap();
    assert_eq!(gradient.shape(), [2, 1, 4]);
    assert_eq!(gradient.stride(), [4, 4, 1]);
    assert_eq!(gradient.storage_offset(), 0);
    assert_eq!(
        gradient
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        SIGMOID_AUTOGRAD_GRADIENT_BITS
    );
    assert_eq!(loss.backward(), Err(TensorError::BackwardGraphFreed));
    assert_eq!(
        leaf.grad()
            .unwrap()
            .unwrap()
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        SIGMOID_AUTOGRAD_GRADIENT_BITS
    );

    let accumulated = Tensor::from_vec(input.clone(), [2, 1, 4])
        .unwrap()
        .with_requires_grad(true);
    for _ in 0..2 {
        accumulated
            .sigmoid()
            .unwrap()
            .mul(&weights)
            .unwrap()
            .sum()
            .backward()
            .unwrap();
    }
    assert_eq!(
        accumulated
            .grad()
            .unwrap()
            .unwrap()
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        SIGMOID_AUTOGRAD_ACCUMULATED_GRADIENT_BITS
    );

    let composed = Tensor::from_vec(input, [2, 1, 4])
        .unwrap()
        .with_requires_grad(true);
    composed
        .sigmoid()
        .unwrap()
        .sin()
        .unwrap()
        .sum()
        .backward()
        .unwrap();
    let expected_composed_gradient = SIGMOID_AUTOGRAD_OUTPUT_BITS.map(|bits| {
        let sigmoid = f32::from_bits(bits);
        (sigmoid.cos() * (1.0 - sigmoid) * sigmoid).to_bits()
    });
    assert_eq!(
        composed
            .grad()
            .unwrap()
            .unwrap()
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        expected_composed_gradient
    );
}

#[test]
fn sigmoid_differentiates_owned_empty_rank_three_tensors() {
    for (shape, expected_stride) in [
        ([0, 1, 3], [3, 3, 1]),
        ([1, 0, 3], [3, 3, 1]),
        ([2, 3, 0], [3, 1, 1]),
        ([0, 0, 0], [1, 1, 1]),
    ] {
        let empty = Tensor::zeros(shape).unwrap().with_requires_grad(true);
        let empty_output = empty.sigmoid().unwrap();
        assert!(empty_output.requires_grad());
        assert!(!empty_output.is_leaf());
        assert_eq!(empty_output.shape(), shape);
        assert_eq!(empty_output.stride(), expected_stride);
        assert_eq!(empty_output.storage_offset(), 0);
        assert!(!empty_output.shares_storage_with(&empty));

        let empty_loss = empty_output.sum();
        empty_loss.backward().unwrap();
        let empty_gradient = empty.grad().unwrap().unwrap();
        assert_eq!(empty_gradient.shape(), shape);
        assert_eq!(empty_gradient.stride(), expected_stride);
        assert_eq!(empty_gradient.storage_offset(), 0);
        assert!(values(&empty_gradient).is_empty());
        assert_eq!(empty_loss.backward(), Err(TensorError::BackwardGraphFreed));
    }
}

#[test]
fn sigmoid_differentiates_owned_rank_four_tensors_with_singletons_and_empty_dimensions() {
    let input = SIGMOID_AUTOGRAD_INPUT_BITS.map(f32::from_bits).to_vec();
    let weights = Tensor::from_vec(
        vec![1.0, -2.0, 0.5, -0.25, 3.0, -4.0, 5.0, -6.0],
        [1, 2, 1, 4],
    )
    .unwrap();
    let leaf = Tensor::from_vec(input.clone(), [1, 2, 1, 4])
        .unwrap()
        .with_requires_grad(true);
    let output = leaf.sigmoid().unwrap();

    assert!(output.requires_grad());
    assert!(!output.is_leaf());
    assert_eq!(output.shape(), [1, 2, 1, 4]);
    assert_eq!(output.stride(), [8, 4, 4, 1]);
    assert_eq!(output.storage_offset(), 0);
    assert_eq!(output.dtype(), leaf.dtype());
    assert_eq!(output.device(), leaf.device());
    assert!(!output.shares_storage_with(&leaf));
    assert_eq!(
        output
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        SIGMOID_AUTOGRAD_OUTPUT_BITS
    );

    let loss = output.mul(&weights).unwrap().sum();
    loss.backward().unwrap();
    let gradient = leaf.grad().unwrap().unwrap();
    assert_eq!(gradient.shape(), [1, 2, 1, 4]);
    assert_eq!(gradient.stride(), [8, 4, 4, 1]);
    assert_eq!(gradient.storage_offset(), 0);
    assert_eq!(gradient.dtype(), leaf.dtype());
    assert_eq!(gradient.device(), leaf.device());
    assert_eq!(
        gradient
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        SIGMOID_AUTOGRAD_GRADIENT_BITS
    );
    assert_eq!(loss.backward(), Err(TensorError::BackwardGraphFreed));
    assert_eq!(
        leaf.grad()
            .unwrap()
            .unwrap()
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        SIGMOID_AUTOGRAD_GRADIENT_BITS
    );

    let accumulated = Tensor::from_vec(input.clone(), [1, 2, 1, 4])
        .unwrap()
        .with_requires_grad(true);
    for _ in 0..2 {
        accumulated
            .sigmoid()
            .unwrap()
            .mul(&weights)
            .unwrap()
            .sum()
            .backward()
            .unwrap();
    }
    assert_eq!(
        accumulated
            .grad()
            .unwrap()
            .unwrap()
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        SIGMOID_AUTOGRAD_ACCUMULATED_GRADIENT_BITS
    );

    let composed = Tensor::from_vec(input, [1, 2, 1, 4])
        .unwrap()
        .with_requires_grad(true);
    composed
        .sigmoid()
        .unwrap()
        .sin()
        .unwrap()
        .sum()
        .backward()
        .unwrap();
    let expected_composed_gradient = SIGMOID_AUTOGRAD_OUTPUT_BITS.map(|bits| {
        let sigmoid = f32::from_bits(bits);
        (sigmoid.cos() * (1.0 - sigmoid) * sigmoid).to_bits()
    });
    assert_eq!(
        composed
            .grad()
            .unwrap()
            .unwrap()
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        expected_composed_gradient
    );
}

#[test]
fn sigmoid_differentiates_owned_empty_rank_four_tensors() {
    for (shape, expected_stride) in [
        ([0, 1, 2, 3], [6, 6, 3, 1]),
        ([1, 0, 2, 3], [6, 6, 3, 1]),
        ([1, 2, 0, 3], [6, 3, 3, 1]),
        ([1, 2, 3, 0], [6, 3, 1, 1]),
        ([0, 0, 0, 0], [1, 1, 1, 1]),
    ] {
        let empty = Tensor::zeros(shape).unwrap().with_requires_grad(true);
        let empty_output = empty.sigmoid().unwrap();
        assert!(empty_output.requires_grad());
        assert!(!empty_output.is_leaf());
        assert_eq!(empty_output.shape(), shape);
        assert_eq!(empty_output.stride(), expected_stride);
        assert_eq!(empty_output.storage_offset(), 0);
        assert_eq!(empty_output.dtype(), empty.dtype());
        assert_eq!(empty_output.device(), empty.device());
        assert!(!empty_output.shares_storage_with(&empty));

        let empty_loss = empty_output.sum();
        empty_loss.backward().unwrap();
        let empty_gradient = empty.grad().unwrap().unwrap();
        assert_eq!(empty_gradient.shape(), shape);
        assert_eq!(empty_gradient.stride(), expected_stride);
        assert_eq!(empty_gradient.storage_offset(), 0);
        assert!(values(&empty_gradient).is_empty());
        assert_eq!(empty_loss.backward(), Err(TensorError::BackwardGraphFreed));
    }
}

#[test]
fn sigmoid_differentiates_owned_rank_five_tensors_with_singletons_and_empty_dimensions() {
    let input = SIGMOID_AUTOGRAD_INPUT_BITS.map(f32::from_bits).to_vec();
    let weights = Tensor::from_vec(
        vec![1.0, -2.0, 0.5, -0.25, 3.0, -4.0, 5.0, -6.0],
        [1, 2, 1, 1, 4],
    )
    .unwrap();
    let leaf = Tensor::from_vec(input.clone(), [1, 2, 1, 1, 4])
        .unwrap()
        .with_requires_grad(true);
    let output = leaf.sigmoid().unwrap();

    assert!(output.requires_grad());
    assert!(!output.is_leaf());
    assert_eq!(output.shape(), [1, 2, 1, 1, 4]);
    assert_eq!(output.stride(), [8, 4, 4, 4, 1]);
    assert_eq!(output.storage_offset(), 0);
    assert_eq!(output.dtype(), leaf.dtype());
    assert_eq!(output.device(), leaf.device());
    assert!(!output.shares_storage_with(&leaf));
    assert_eq!(
        output
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        SIGMOID_AUTOGRAD_OUTPUT_BITS
    );

    let loss = output.mul(&weights).unwrap().sum();
    loss.backward().unwrap();
    let gradient = leaf.grad().unwrap().unwrap();
    assert_eq!(gradient.shape(), [1, 2, 1, 1, 4]);
    assert_eq!(gradient.stride(), [8, 4, 4, 4, 1]);
    assert_eq!(gradient.storage_offset(), 0);
    assert_eq!(gradient.dtype(), leaf.dtype());
    assert_eq!(gradient.device(), leaf.device());
    assert_eq!(
        gradient
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        SIGMOID_AUTOGRAD_GRADIENT_BITS
    );
    assert_eq!(loss.backward(), Err(TensorError::BackwardGraphFreed));
    assert_eq!(
        leaf.grad()
            .unwrap()
            .unwrap()
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        SIGMOID_AUTOGRAD_GRADIENT_BITS
    );

    let accumulated = Tensor::from_vec(input.clone(), [1, 2, 1, 1, 4])
        .unwrap()
        .with_requires_grad(true);
    for _ in 0..2 {
        accumulated
            .sigmoid()
            .unwrap()
            .mul(&weights)
            .unwrap()
            .sum()
            .backward()
            .unwrap();
    }
    assert_eq!(
        accumulated
            .grad()
            .unwrap()
            .unwrap()
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        SIGMOID_AUTOGRAD_ACCUMULATED_GRADIENT_BITS
    );

    let composed = Tensor::from_vec(input, [1, 2, 1, 1, 4])
        .unwrap()
        .with_requires_grad(true);
    composed
        .sigmoid()
        .unwrap()
        .sin()
        .unwrap()
        .sum()
        .backward()
        .unwrap();
    let expected_composed_gradient = SIGMOID_AUTOGRAD_OUTPUT_BITS.map(|bits| {
        let sigmoid = f32::from_bits(bits);
        (sigmoid.cos() * (1.0 - sigmoid) * sigmoid).to_bits()
    });
    assert_eq!(
        composed
            .grad()
            .unwrap()
            .unwrap()
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        expected_composed_gradient
    );
}

#[test]
fn sigmoid_differentiates_owned_empty_rank_five_tensors() {
    for (shape, expected_stride) in [
        ([0, 1, 2, 3, 4], [24, 24, 12, 4, 1]),
        ([1, 0, 2, 3, 4], [24, 24, 12, 4, 1]),
        ([1, 2, 0, 3, 4], [24, 12, 12, 4, 1]),
        ([1, 2, 3, 0, 4], [24, 12, 4, 4, 1]),
        ([1, 2, 3, 4, 0], [24, 12, 4, 1, 1]),
        ([0, 0, 0, 0, 0], [1, 1, 1, 1, 1]),
    ] {
        let empty = Tensor::zeros(shape).unwrap().with_requires_grad(true);
        let empty_output = empty.sigmoid().unwrap();
        assert!(empty_output.requires_grad());
        assert!(!empty_output.is_leaf());
        assert_eq!(empty_output.shape(), shape);
        assert_eq!(empty_output.stride(), expected_stride);
        assert_eq!(empty_output.storage_offset(), 0);
        assert_eq!(empty_output.dtype(), empty.dtype());
        assert_eq!(empty_output.device(), empty.device());
        assert!(!empty_output.shares_storage_with(&empty));

        let empty_loss = empty_output.sum();
        empty_loss.backward().unwrap();
        let empty_gradient = empty.grad().unwrap().unwrap();
        assert_eq!(empty_gradient.shape(), shape);
        assert_eq!(empty_gradient.stride(), expected_stride);
        assert_eq!(empty_gradient.storage_offset(), 0);
        assert!(values(&empty_gradient).is_empty());
        assert_eq!(empty_loss.backward(), Err(TensorError::BackwardGraphFreed));
    }
}

#[test]
fn sigmoid_scalar_autograd_composes_accumulates_and_obeys_grad_mode() {
    let composed = Tensor::from_vec(vec![0.5], [])
        .unwrap()
        .with_requires_grad(true);
    composed
        .sigmoid()
        .unwrap()
        .sin()
        .unwrap()
        .backward()
        .unwrap();
    let sigmoid = 1.0_f32 / (1.0 + (-0.5_f32).exp());
    assert_eq!(
        composed.grad().unwrap().unwrap().item().unwrap().to_bits(),
        (sigmoid.cos() * (1.0 - sigmoid) * sigmoid).to_bits()
    );

    let accumulated = Tensor::from_vec(vec![-0.5], [])
        .unwrap()
        .with_requires_grad(true);
    accumulated.sigmoid().unwrap().backward().unwrap();
    let first = accumulated.grad().unwrap().unwrap().item().unwrap();
    accumulated.sigmoid().unwrap().backward().unwrap();
    assert_eq!(
        accumulated
            .grad()
            .unwrap()
            .unwrap()
            .item()
            .unwrap()
            .to_bits(),
        (first * 2.0).to_bits()
    );

    {
        let _guard = no_grad();
        let output = accumulated.sigmoid().unwrap();
        assert!(!output.requires_grad());
        assert!(output.is_leaf());
        assert!(!output.shares_storage_with(&accumulated));
    }
    assert!(accumulated.sigmoid().unwrap().requires_grad());

    let detached = accumulated.detach().unwrap().sigmoid().unwrap();
    assert!(!detached.requires_grad());
    assert!(detached.is_leaf());
    assert!(!detached.shares_storage_with(&accumulated));
}

#[test]
fn sigmoid_rejects_nonfinite_and_rank_six_owned_leaves_before_graph_mutation() {
    let unsupported = TensorError::AutogradRecordingUnsupported {
        operation: "sigmoid",
    };

    for bits in [
        f32::INFINITY.to_bits(),
        f32::NEG_INFINITY.to_bits(),
        0x7f81_2345,
        0xffc5_4321,
    ] {
        let value = f32::from_bits(bits);
        let leaf = Tensor::from_vec(vec![value], [])
            .unwrap()
            .with_requires_grad(true);
        assert_eq!(leaf.sigmoid(), Err(unsupported.clone()));
        assert!(leaf.grad().unwrap().is_none());
        leaf.sum().backward().unwrap();
        assert_eq!(
            leaf.grad().unwrap().unwrap().item().unwrap().to_bits(),
            1.0_f32.to_bits()
        );

        let vector = Tensor::from_vec(vec![0.5, value], [2])
            .unwrap()
            .with_requires_grad(true);
        assert_eq!(vector.sigmoid(), Err(unsupported.clone()));
        assert!(vector.grad().unwrap().is_none());
        vector.sum().backward().unwrap();
        assert_eq!(values(&vector.grad().unwrap().unwrap()), [1.0, 1.0]);

        let matrix = Tensor::from_vec(vec![0.5, value], [1, 2])
            .unwrap()
            .with_requires_grad(true);
        assert_eq!(matrix.sigmoid(), Err(unsupported.clone()));
        assert!(matrix.grad().unwrap().is_none());
        matrix.sum().backward().unwrap();
        assert_eq!(values(&matrix.grad().unwrap().unwrap()), [1.0, 1.0]);

        let rank_three = Tensor::from_vec(vec![0.5, value], [1, 1, 2])
            .unwrap()
            .with_requires_grad(true);
        assert_eq!(rank_three.sigmoid(), Err(unsupported.clone()));
        assert!(rank_three.grad().unwrap().is_none());
        rank_three.sum().backward().unwrap();
        assert_eq!(values(&rank_three.grad().unwrap().unwrap()), [1.0, 1.0]);

        let rank_four = Tensor::from_vec(vec![0.5, value], [1, 1, 1, 2])
            .unwrap()
            .with_requires_grad(true);
        assert_eq!(rank_four.sigmoid(), Err(unsupported.clone()));
        assert!(rank_four.grad().unwrap().is_none());
        rank_four.sum().backward().unwrap();
        assert_eq!(values(&rank_four.grad().unwrap().unwrap()), [1.0, 1.0]);

        let rank_five = Tensor::from_vec(vec![0.5, value], [1, 1, 1, 1, 2])
            .unwrap()
            .with_requires_grad(true);
        assert_eq!(rank_five.sigmoid(), Err(unsupported.clone()));
        assert!(rank_five.grad().unwrap().is_none());
        rank_five.sum().backward().unwrap();
        assert_eq!(values(&rank_five.grad().unwrap().unwrap()), [1.0, 1.0]);
    }

    let rank_six = Tensor::from_vec(vec![0.5, -1.0], [1, 1, 1, 1, 1, 2])
        .unwrap()
        .with_requires_grad(true);
    assert_eq!(rank_six.sigmoid(), Err(unsupported.clone()));
    assert!(rank_six.grad().unwrap().is_none());
    rank_six.sum().backward().unwrap();
    assert_eq!(values(&rank_six.grad().unwrap().unwrap()), [1.0, 1.0]);

    let extreme = Tensor::zeros([0])
        .unwrap()
        .reshape([0, i64::MAX, 1, 1, 1, 3])
        .unwrap()
        .with_requires_grad(true);
    assert_eq!(extreme.sigmoid(), Err(unsupported));
    {
        let _guard = no_grad();
        assert_eq!(
            extreme.sigmoid(),
            Err(TensorError::StrideCalculationOverflow)
        );
    }
}

#[test]
fn sigmoid_rejects_tracked_views_and_nonleaves_before_graph_mutation() {
    let unsupported = TensorError::AutogradRecordingUnsupported {
        operation: "sigmoid",
    };

    let view_base = Tensor::from_vec(vec![0.5], [1])
        .unwrap()
        .with_requires_grad(true);
    let scalar_view = view_base.index([0]).unwrap();
    assert!(scalar_view.requires_grad());
    assert!(!scalar_view.is_leaf());
    assert_eq!(scalar_view.sigmoid(), Err(unsupported.clone()));
    scalar_view.backward().unwrap();
    assert_eq!(values(&view_base.grad().unwrap().unwrap()), [1.0]);

    let vector_view_base = Tensor::from_vec(vec![0.5, -1.0, 2.0, -3.0], [2, 2])
        .unwrap()
        .with_requires_grad(true);
    let vector_view = vector_view_base.index([0]).unwrap();
    assert!(vector_view.requires_grad());
    assert!(!vector_view.is_leaf());
    assert_eq!(vector_view.shape(), [2]);
    assert_eq!(vector_view.sigmoid(), Err(unsupported.clone()));
    vector_view.sum().backward().unwrap();
    assert_eq!(
        values(&vector_view_base.grad().unwrap().unwrap()),
        [1.0, 1.0, 0.0, 0.0]
    );

    let matrix_view_base =
        Tensor::from_vec(vec![0.5, -1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0], [2, 2, 2])
            .unwrap()
            .with_requires_grad(true);
    let matrix_view = matrix_view_base.index([0]).unwrap();
    assert!(matrix_view.requires_grad());
    assert!(!matrix_view.is_leaf());
    assert_eq!(matrix_view.shape(), [2, 2]);
    assert_eq!(matrix_view.sigmoid(), Err(unsupported.clone()));
    matrix_view.sum().backward().unwrap();
    assert_eq!(
        values(&matrix_view_base.grad().unwrap().unwrap()),
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    );

    let nonleaf_base = Tensor::from_vec(vec![0.5], [])
        .unwrap()
        .with_requires_grad(true);
    let nonleaf = nonleaf_base.sin().unwrap();
    assert_eq!(nonleaf.sigmoid(), Err(unsupported.clone()));
    nonleaf.backward().unwrap();
    assert_eq!(
        nonleaf_base
            .grad()
            .unwrap()
            .unwrap()
            .item()
            .unwrap()
            .to_bits(),
        0.5_f32.cos().to_bits()
    );

    let matrix_nonleaf_base = Tensor::from_vec(vec![0.5, -0.5], [1, 2])
        .unwrap()
        .with_requires_grad(true);
    let matrix_nonleaf = matrix_nonleaf_base.sin().unwrap();
    assert_eq!(matrix_nonleaf.sigmoid(), Err(unsupported.clone()));
    matrix_nonleaf.sum().backward().unwrap();
    assert_eq!(
        matrix_nonleaf_base
            .grad()
            .unwrap()
            .unwrap()
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        [0.5_f32.cos().to_bits(), (-0.5_f32).cos().to_bits()]
    );

    let no_grad_view = {
        let _guard = no_grad();
        vector_view_base.index([0]).unwrap()
    };
    assert!(no_grad_view.requires_grad());
    assert!(no_grad_view.is_leaf());
    assert_eq!(no_grad_view.shape(), [2]);
    assert_eq!(no_grad_view.sigmoid(), Err(unsupported.clone()));

    let empty_view_base = Tensor::zeros([1, 0]).unwrap().with_requires_grad(true);
    let empty_view = {
        let _guard = no_grad();
        empty_view_base.index([0]).unwrap()
    };
    assert!(empty_view.requires_grad());
    assert!(empty_view.is_leaf());
    assert_eq!(empty_view.shape(), [0]);
    assert_eq!(empty_view.sigmoid(), Err(unsupported));
}

#[test]
fn sigmoid_rejects_rank_four_tracked_views_and_nonleaves() {
    let unsupported = TensorError::AutogradRecordingUnsupported {
        operation: "sigmoid",
    };

    let view_base = Tensor::from_vec(vec![0.5, -1.0, 2.0, -3.0], [2, 1, 1, 1, 2])
        .unwrap()
        .with_requires_grad(true);
    let view = view_base.index([0]).unwrap();
    assert!(view.requires_grad());
    assert!(!view.is_leaf());
    assert_eq!(view.shape(), [1, 1, 1, 2]);
    assert_eq!(view.sigmoid(), Err(unsupported.clone()));
    view.sum().backward().unwrap();
    assert_eq!(
        values(&view_base.grad().unwrap().unwrap()),
        [1.0, 1.0, 0.0, 0.0]
    );

    let nonleaf_base = Tensor::from_vec(vec![0.5, -0.5], [1, 1, 1, 2])
        .unwrap()
        .with_requires_grad(true);
    let nonleaf = nonleaf_base.sin().unwrap();
    assert_eq!(nonleaf.sigmoid(), Err(unsupported));
    nonleaf.sum().backward().unwrap();
    assert_eq!(
        nonleaf_base
            .grad()
            .unwrap()
            .unwrap()
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        [0.5_f32.cos().to_bits(), (-0.5_f32).cos().to_bits()]
    );
}

#[test]
fn sigmoid_rejects_rank_five_tracked_views_and_nonleaves() {
    let unsupported = TensorError::AutogradRecordingUnsupported {
        operation: "sigmoid",
    };

    let view_base = Tensor::from_vec(vec![0.5, -1.0, 2.0, -3.0], [2, 1, 1, 1, 1, 2])
        .unwrap()
        .with_requires_grad(true);
    let view = view_base.index([0]).unwrap();
    assert!(view.requires_grad());
    assert!(!view.is_leaf());
    assert_eq!(view.shape(), [1, 1, 1, 1, 2]);
    assert_eq!(view.sigmoid(), Err(unsupported.clone()));
    view.sum().backward().unwrap();
    assert_eq!(
        values(&view_base.grad().unwrap().unwrap()),
        [1.0, 1.0, 0.0, 0.0]
    );

    let nonleaf_base = Tensor::from_vec(vec![0.5, -0.5], [1, 1, 1, 1, 2])
        .unwrap()
        .with_requires_grad(true);
    let nonleaf = nonleaf_base.sin().unwrap();
    assert_eq!(nonleaf.sigmoid(), Err(unsupported));
    nonleaf.sum().backward().unwrap();
    assert_eq!(
        nonleaf_base
            .grad()
            .unwrap()
            .unwrap()
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        [0.5_f32.cos().to_bits(), (-0.5_f32).cos().to_bits()]
    );
}

#[test]
fn tanh_differentiates_finite_owned_scalars_at_signed_zero_and_saturation() {
    // input, forward result, unit-upstream gradient
    const CASES: [(u32, u32, u32); 10] = [
        (0x0000_0000, 0x0000_0000, 0x3f80_0000),
        (0x8000_0000, 0x8000_0000, 0x3f80_0000),
        (0x0000_0001, 0x0000_0001, 0x3f80_0000),
        (0x8000_0001, 0x8000_0001, 0x3f80_0000),
        (0x3f00_0000, 0x3eec_9a9f, 0x3f49_54a3),
        (0xbf00_0000, 0xbeec_9a9f, 0x3f49_54a3),
        (0x4110_2c66, 0x3f7f_ffff, 0x3400_0000),
        (0xc110_2c66, 0xbf7f_ffff, 0x3400_0000),
        (0x4110_2c67, 0x3f80_0000, 0x0000_0000),
        (0xc110_2c67, 0xbf80_0000, 0x0000_0000),
    ];

    for (input_bits, output_bits, gradient_bits) in CASES {
        let leaf = Tensor::from_vec(vec![f32::from_bits(input_bits)], [])
            .unwrap()
            .with_requires_grad(true);
        let output = leaf.tanh().unwrap();

        assert!(output.requires_grad());
        assert!(!output.is_leaf());
        assert!(output.shape().is_empty());
        assert!(output.stride().is_empty());
        assert_eq!(output.storage_offset(), 0);
        assert!(!output.shares_storage_with(&leaf));
        assert_eq!(output.item().unwrap().to_bits(), output_bits);

        output.backward().unwrap();
        assert_eq!(
            leaf.grad().unwrap().unwrap().item().unwrap().to_bits(),
            gradient_bits
        );
        assert_eq!(output.backward(), Err(TensorError::BackwardGraphFreed));
    }
}

#[test]
fn tanh_differentiates_owned_vectors_with_weighting_accumulation_and_graph_freeing() {
    let input = TANH_AUTOGRAD_INPUT_BITS.map(f32::from_bits).to_vec();
    let weights = Tensor::from_vec(vec![1.0, -2.0, 0.5, -0.25, 3.0, -4.0, 5.0, -6.0], [8]).unwrap();
    let leaf = Tensor::from_vec(input.clone(), [8])
        .unwrap()
        .with_requires_grad(true);
    let output = leaf.tanh().unwrap();

    assert!(output.requires_grad());
    assert!(!output.is_leaf());
    assert_eq!(output.shape(), [8]);
    assert_eq!(output.stride(), [1]);
    assert_eq!(output.storage_offset(), 0);
    assert!(!output.shares_storage_with(&leaf));
    assert_eq!(
        output
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        TANH_AUTOGRAD_OUTPUT_BITS
    );

    let loss = output.mul(&weights).unwrap().sum();
    loss.backward().unwrap();
    let gradient_bits = leaf
        .grad()
        .unwrap()
        .unwrap()
        .logical_values()
        .map(f32::to_bits)
        .collect::<Vec<_>>();
    assert_eq!(gradient_bits, TANH_AUTOGRAD_GRADIENT_BITS);
    assert_eq!(loss.backward(), Err(TensorError::BackwardGraphFreed));
    assert_eq!(
        leaf.grad()
            .unwrap()
            .unwrap()
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        TANH_AUTOGRAD_GRADIENT_BITS
    );

    let accumulated = Tensor::from_vec(input, [8])
        .unwrap()
        .with_requires_grad(true);
    for _ in 0..2 {
        accumulated
            .tanh()
            .unwrap()
            .mul(&weights)
            .unwrap()
            .sum()
            .backward()
            .unwrap();
    }
    assert_eq!(
        accumulated
            .grad()
            .unwrap()
            .unwrap()
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        TANH_AUTOGRAD_ACCUMULATED_GRADIENT_BITS
    );

    let composed = Tensor::from_vec(vec![0.5, -0.5], [2])
        .unwrap()
        .with_requires_grad(true);
    composed
        .tanh()
        .unwrap()
        .sin()
        .unwrap()
        .sum()
        .backward()
        .unwrap();
    assert_eq!(
        composed
            .grad()
            .unwrap()
            .unwrap()
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        [0x3f34_3692, 0x3f34_3692]
    );

    let empty = Tensor::zeros([0]).unwrap().with_requires_grad(true);
    let empty_output = empty.tanh().unwrap();
    assert!(empty_output.requires_grad());
    assert!(!empty_output.is_leaf());
    assert_eq!(empty_output.shape(), [0]);
    assert_eq!(empty_output.stride(), [1]);
    assert!(!empty_output.shares_storage_with(&empty));
    let empty_loss = empty_output.sum();
    empty_loss.backward().unwrap();
    let empty_gradient = empty.grad().unwrap().unwrap();
    assert_eq!(empty_gradient.shape(), [0]);
    assert_eq!(empty_gradient.stride(), [1]);
    assert!(values(&empty_gradient).is_empty());
    assert_eq!(empty_loss.backward(), Err(TensorError::BackwardGraphFreed));
}

#[test]
fn tanh_differentiates_owned_matrices_with_weighting_accumulation_and_graph_freeing() {
    let input = TANH_AUTOGRAD_INPUT_BITS.map(f32::from_bits).to_vec();
    let weights =
        Tensor::from_vec(vec![1.0, -2.0, 0.5, -0.25, 3.0, -4.0, 5.0, -6.0], [2, 4]).unwrap();
    let leaf = Tensor::from_vec(input.clone(), [2, 4])
        .unwrap()
        .with_requires_grad(true);
    let output = leaf.tanh().unwrap();

    assert!(output.requires_grad());
    assert!(!output.is_leaf());
    assert_eq!(output.shape(), [2, 4]);
    assert_eq!(output.stride(), [4, 1]);
    assert_eq!(output.storage_offset(), 0);
    assert!(!output.shares_storage_with(&leaf));
    assert_eq!(
        output
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        TANH_AUTOGRAD_OUTPUT_BITS
    );

    let loss = output.mul(&weights).unwrap().sum();
    loss.backward().unwrap();
    let gradient = leaf.grad().unwrap().unwrap();
    assert_eq!(gradient.shape(), [2, 4]);
    assert_eq!(gradient.stride(), [4, 1]);
    assert_eq!(
        gradient
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        TANH_AUTOGRAD_GRADIENT_BITS
    );
    assert_eq!(loss.backward(), Err(TensorError::BackwardGraphFreed));
    assert_eq!(
        leaf.grad()
            .unwrap()
            .unwrap()
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        TANH_AUTOGRAD_GRADIENT_BITS
    );

    let accumulated = Tensor::from_vec(input, [2, 4])
        .unwrap()
        .with_requires_grad(true);
    for _ in 0..2 {
        accumulated
            .tanh()
            .unwrap()
            .mul(&weights)
            .unwrap()
            .sum()
            .backward()
            .unwrap();
    }
    assert_eq!(
        accumulated
            .grad()
            .unwrap()
            .unwrap()
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        TANH_AUTOGRAD_ACCUMULATED_GRADIENT_BITS
    );

    for (shape, expected_stride) in [([0, 0], [1, 1]), ([0, 3], [3, 1]), ([2, 0], [1, 1])] {
        let empty = Tensor::zeros(shape).unwrap().with_requires_grad(true);
        let empty_output = empty.tanh().unwrap();
        assert!(empty_output.requires_grad());
        assert!(!empty_output.is_leaf());
        assert_eq!(empty_output.shape(), shape);
        assert_eq!(empty_output.stride(), expected_stride);
        assert_eq!(empty_output.storage_offset(), 0);
        assert!(!empty_output.shares_storage_with(&empty));
        let empty_loss = empty_output.sum();
        empty_loss.backward().unwrap();
        let empty_gradient = empty.grad().unwrap().unwrap();
        assert_eq!(empty_gradient.shape(), shape);
        assert_eq!(empty_gradient.stride(), expected_stride);
        assert!(values(&empty_gradient).is_empty());
        assert_eq!(empty_loss.backward(), Err(TensorError::BackwardGraphFreed));
    }
}

#[test]
fn tanh_differentiates_owned_rank_three_tensors_with_singletons_and_empty_dimensions() {
    let input = TANH_AUTOGRAD_INPUT_BITS.map(f32::from_bits).to_vec();
    let weights =
        Tensor::from_vec(vec![1.0, -2.0, 0.5, -0.25, 3.0, -4.0, 5.0, -6.0], [2, 1, 4]).unwrap();
    let leaf = Tensor::from_vec(input.clone(), [2, 1, 4])
        .unwrap()
        .with_requires_grad(true);
    let output = leaf.tanh().unwrap();

    assert!(output.requires_grad());
    assert!(!output.is_leaf());
    assert_eq!(output.shape(), [2, 1, 4]);
    assert_eq!(output.stride(), [4, 4, 1]);
    assert_eq!(output.storage_offset(), 0);
    assert!(!output.shares_storage_with(&leaf));
    assert_eq!(
        output
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        TANH_AUTOGRAD_OUTPUT_BITS
    );

    let loss = output.mul(&weights).unwrap().sum();
    loss.backward().unwrap();
    let gradient = leaf.grad().unwrap().unwrap();
    assert_eq!(gradient.shape(), [2, 1, 4]);
    assert_eq!(gradient.stride(), [4, 4, 1]);
    assert_eq!(
        gradient
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        TANH_AUTOGRAD_GRADIENT_BITS
    );
    assert_eq!(loss.backward(), Err(TensorError::BackwardGraphFreed));
    assert_eq!(
        leaf.grad()
            .unwrap()
            .unwrap()
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        TANH_AUTOGRAD_GRADIENT_BITS
    );

    let accumulated = Tensor::from_vec(input, [2, 1, 4])
        .unwrap()
        .with_requires_grad(true);
    for _ in 0..2 {
        accumulated
            .tanh()
            .unwrap()
            .mul(&weights)
            .unwrap()
            .sum()
            .backward()
            .unwrap();
    }
    assert_eq!(
        accumulated
            .grad()
            .unwrap()
            .unwrap()
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        TANH_AUTOGRAD_ACCUMULATED_GRADIENT_BITS
    );

    for (shape, expected_stride) in [
        ([0, 1, 4], [4, 4, 1]),
        ([2, 0, 4], [4, 4, 1]),
        ([2, 1, 0], [1, 1, 1]),
        ([1, 0, 1], [1, 1, 1]),
    ] {
        let empty = Tensor::zeros(shape).unwrap().with_requires_grad(true);
        let empty_output = empty.tanh().unwrap();
        assert!(empty_output.requires_grad());
        assert!(!empty_output.is_leaf());
        assert_eq!(empty_output.shape(), shape);
        assert_eq!(empty_output.stride(), expected_stride);
        assert_eq!(empty_output.storage_offset(), 0);
        assert!(!empty_output.shares_storage_with(&empty));
        let empty_loss = empty_output.sum();
        empty_loss.backward().unwrap();
        let empty_gradient = empty.grad().unwrap().unwrap();
        assert_eq!(empty_gradient.shape(), shape);
        assert_eq!(empty_gradient.stride(), expected_stride);
        assert!(values(&empty_gradient).is_empty());
        assert_eq!(empty_loss.backward(), Err(TensorError::BackwardGraphFreed));
    }
}

#[test]
fn tanh_scalar_autograd_composes_accumulates_and_obeys_grad_mode() {
    let composed = Tensor::from_vec(vec![0.5], [])
        .unwrap()
        .with_requires_grad(true);
    composed.tanh().unwrap().sin().unwrap().backward().unwrap();
    let tanh = 0.5_f32.tanh();
    assert_eq!(
        composed.grad().unwrap().unwrap().item().unwrap().to_bits(),
        (tanh.cos() * (-tanh).mul_add(tanh, 1.0)).to_bits()
    );

    let accumulated = Tensor::from_vec(vec![-0.5], [])
        .unwrap()
        .with_requires_grad(true);
    accumulated.tanh().unwrap().backward().unwrap();
    let first = accumulated.grad().unwrap().unwrap().item().unwrap();
    accumulated.tanh().unwrap().backward().unwrap();
    assert_eq!(
        accumulated
            .grad()
            .unwrap()
            .unwrap()
            .item()
            .unwrap()
            .to_bits(),
        (first * 2.0).to_bits()
    );

    {
        let _guard = no_grad();
        let output = accumulated.tanh().unwrap();
        assert!(!output.requires_grad());
        assert!(output.is_leaf());
        assert!(!output.shares_storage_with(&accumulated));
    }
    assert!(accumulated.tanh().unwrap().requires_grad());

    let detached = accumulated.detach().unwrap().tanh().unwrap();
    assert!(!detached.requires_grad());
    assert!(detached.is_leaf());
    assert!(!detached.shares_storage_with(&accumulated));
}

#[test]
fn tanh_rejects_nonfinite_and_rank_four_owned_leaves_before_graph_mutation() {
    let unsupported = TensorError::AutogradRecordingUnsupported { operation: "tanh" };

    for bits in [
        f32::INFINITY.to_bits(),
        f32::NEG_INFINITY.to_bits(),
        0x7f81_2345,
        0xffc5_4321,
    ] {
        let leaf = Tensor::from_vec(vec![f32::from_bits(bits)], [])
            .unwrap()
            .with_requires_grad(true);
        assert_eq!(leaf.tanh(), Err(unsupported.clone()));
        assert!(leaf.grad().unwrap().is_none());
        leaf.sum().backward().unwrap();
        assert_eq!(
            leaf.grad().unwrap().unwrap().item().unwrap().to_bits(),
            1.0_f32.to_bits()
        );

        let vector = Tensor::from_vec(vec![0.5, f32::from_bits(bits)], [2])
            .unwrap()
            .with_requires_grad(true);
        assert_eq!(vector.tanh(), Err(unsupported.clone()));
        assert!(vector.grad().unwrap().is_none());
        vector.sum().backward().unwrap();
        assert_eq!(values(&vector.grad().unwrap().unwrap()), [1.0, 1.0]);

        let matrix = Tensor::from_vec(vec![0.5, f32::from_bits(bits)], [1, 2])
            .unwrap()
            .with_requires_grad(true);
        assert_eq!(matrix.tanh(), Err(unsupported.clone()));
        assert!(matrix.grad().unwrap().is_none());
        matrix.sum().backward().unwrap();
        assert_eq!(values(&matrix.grad().unwrap().unwrap()), [1.0, 1.0]);

        let rank_three = Tensor::from_vec(vec![0.5, f32::from_bits(bits)], [1, 1, 2])
            .unwrap()
            .with_requires_grad(true);
        assert_eq!(rank_three.tanh(), Err(unsupported.clone()));
        assert!(rank_three.grad().unwrap().is_none());
        rank_three.sum().backward().unwrap();
        assert_eq!(values(&rank_three.grad().unwrap().unwrap()), [1.0, 1.0]);
    }

    let rank_four = Tensor::from_vec(vec![0.5, -1.0], [1, 1, 1, 2])
        .unwrap()
        .with_requires_grad(true);
    assert_eq!(rank_four.tanh(), Err(unsupported.clone()));
    assert!(rank_four.grad().unwrap().is_none());
    rank_four.sum().backward().unwrap();
    assert_eq!(values(&rank_four.grad().unwrap().unwrap()), [1.0, 1.0]);
}

#[test]
fn tanh_rejects_tracked_views_and_nonleaves_before_graph_or_layout_mutation() {
    let unsupported = TensorError::AutogradRecordingUnsupported { operation: "tanh" };

    let view_base = Tensor::from_vec(vec![0.5], [1])
        .unwrap()
        .with_requires_grad(true);
    let scalar_view = view_base.index([0]).unwrap();
    assert!(scalar_view.requires_grad());
    assert!(!scalar_view.is_leaf());
    assert_eq!(scalar_view.tanh(), Err(unsupported.clone()));
    scalar_view.backward().unwrap();
    assert_eq!(values(&view_base.grad().unwrap().unwrap()), [1.0]);

    let vector_view_base = Tensor::from_vec(vec![0.5, -1.0, 2.0, -3.0], [2, 2])
        .unwrap()
        .with_requires_grad(true);
    let vector_view = vector_view_base.index([0]).unwrap();
    assert!(vector_view.requires_grad());
    assert!(!vector_view.is_leaf());
    assert_eq!(vector_view.shape(), [2]);
    assert_eq!(vector_view.tanh(), Err(unsupported.clone()));
    vector_view.sum().backward().unwrap();
    assert_eq!(
        values(&vector_view_base.grad().unwrap().unwrap()),
        [1.0, 1.0, 0.0, 0.0]
    );

    let matrix_view_base =
        Tensor::from_vec(vec![0.5, -1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0], [2, 2, 2])
            .unwrap()
            .with_requires_grad(true);
    let matrix_view = matrix_view_base.index([0]).unwrap();
    assert!(matrix_view.requires_grad());
    assert!(!matrix_view.is_leaf());
    assert_eq!(matrix_view.shape(), [2, 2]);
    assert_eq!(matrix_view.tanh(), Err(unsupported.clone()));
    matrix_view.sum().backward().unwrap();
    assert_eq!(
        values(&matrix_view_base.grad().unwrap().unwrap()),
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    );

    let rank_three_view_base = Tensor::from_vec(vec![0.5, -1.0, 2.0, -3.0], [2, 1, 1, 2])
        .unwrap()
        .with_requires_grad(true);
    let rank_three_view = rank_three_view_base.index([0]).unwrap();
    assert!(rank_three_view.requires_grad());
    assert!(!rank_three_view.is_leaf());
    assert_eq!(rank_three_view.shape(), [1, 1, 2]);
    assert_eq!(rank_three_view.tanh(), Err(unsupported.clone()));
    rank_three_view.sum().backward().unwrap();
    assert_eq!(
        values(&rank_three_view_base.grad().unwrap().unwrap()),
        [1.0, 1.0, 0.0, 0.0]
    );

    let nonleaf_base = Tensor::from_vec(vec![0.5, -0.5], [1, 1, 2])
        .unwrap()
        .with_requires_grad(true);
    let nonleaf = nonleaf_base.sin().unwrap();
    assert_eq!(nonleaf.tanh(), Err(unsupported.clone()));
    nonleaf.sum().backward().unwrap();
    assert_eq!(
        values(&nonleaf_base.grad().unwrap().unwrap()),
        [0.5_f32.cos(), (-0.5_f32).cos()]
    );

    let no_grad_view = {
        let _guard = no_grad();
        matrix_view_base.index([0]).unwrap()
    };
    assert!(no_grad_view.requires_grad());
    assert!(no_grad_view.is_leaf());
    assert_eq!(no_grad_view.shape(), [2, 2]);
    assert_eq!(no_grad_view.tanh(), Err(unsupported.clone()));

    let no_grad_rank_three_view = {
        let _guard = no_grad();
        rank_three_view_base.index([0]).unwrap()
    };
    assert!(no_grad_rank_three_view.requires_grad());
    assert!(no_grad_rank_three_view.is_leaf());
    assert_eq!(no_grad_rank_three_view.shape(), [1, 1, 2]);
    assert_eq!(no_grad_rank_three_view.tanh(), Err(unsupported.clone()));

    let empty_view_base = Tensor::zeros([1, 0]).unwrap().with_requires_grad(true);
    let empty_view = {
        let _guard = no_grad();
        empty_view_base.index([0]).unwrap()
    };
    assert!(empty_view.requires_grad());
    assert!(empty_view.is_leaf());
    assert_eq!(empty_view.shape(), [0]);
    assert_eq!(empty_view.tanh(), Err(unsupported.clone()));

    let extreme = Tensor::zeros([0])
        .unwrap()
        .reshape([0, 1, i64::MAX, 3])
        .unwrap()
        .with_requires_grad(true);
    assert_eq!(extreme.tanh(), Err(unsupported));
    {
        let _guard = no_grad();
        assert_eq!(extreme.tanh(), Err(TensorError::StrideCalculationOverflow));
    }
}

#[test]
fn leaf_gradient_snapshots_preserve_the_contiguous_slice_contract() {
    let leaf = Tensor::from_vec(vec![2.0, 3.0], [2])
        .unwrap()
        .with_requires_grad(true);
    let loss = leaf.sum();

    loss.backward().unwrap();
    let retained = leaf.grad().unwrap().unwrap();
    let current = leaf.grad().unwrap().unwrap();
    assert!(!retained.shares_storage_with(&current));
    assert_eq!(retained.as_slice(), [1.0, 1.0]);
    assert_eq!(current.as_slice(), [1.0, 1.0]);

    loss.backward().unwrap();
    assert_eq!(retained.as_slice(), [1.0, 1.0]);
    assert_eq!(leaf.grad().unwrap().unwrap().as_slice(), [2.0, 2.0]);
}

#[test]
fn item_does_not_mutate_a_one_element_view_graph() {
    let leaf = Tensor::from_vec(
        [0x0000_0000, 0x7fc1_2345].map(f32::from_bits).to_vec(),
        [1, 2],
    )
    .unwrap()
    .with_requires_grad(true);
    let view = leaf.transpose(0, 1).unwrap().index([1]).unwrap();

    assert!(view.requires_grad());
    assert!(!view.is_leaf());
    assert!(leaf.grad().unwrap().is_none());
    assert_eq!(view.item().unwrap().to_bits(), 0x7fc1_2345);
    assert!(view.requires_grad());
    assert!(!view.is_leaf());
    assert!(leaf.grad().unwrap().is_none());

    view.sum().backward().unwrap();
    assert_eq!(
        leaf.grad()
            .unwrap()
            .unwrap()
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        [0x0000_0000, 0x3f80_0000]
    );
}

#[test]
fn concurrent_backward_on_shared_graph_commits_one_complete_traversal() {
    let leaf = Tensor::from_vec(vec![3.0], [])
        .unwrap()
        .with_requires_grad(true);
    let mut forward = leaf.mul_scalar(1.0).unwrap();
    let mut reverse = leaf.mul_scalar(1.0).unwrap();
    for _ in 0..2_000 {
        forward = forward.mul_scalar(1.0).unwrap();
        reverse = reverse.mul_scalar(1.0).unwrap();
    }
    let forward_root = forward.mul(&reverse).unwrap().sum();
    let reverse_root = reverse.mul(&forward).unwrap().sum();
    let barrier = Arc::new(Barrier::new(3));

    let (forward_result, reverse_result) = thread::scope(|scope| {
        let forward_barrier = Arc::clone(&barrier);
        let forward_thread = scope.spawn(move || {
            forward_barrier.wait();
            forward_root.backward()
        });
        let reverse_barrier = Arc::clone(&barrier);
        let reverse_thread = scope.spawn(move || {
            reverse_barrier.wait();
            reverse_root.backward()
        });
        barrier.wait();
        (
            forward_thread.join().unwrap(),
            reverse_thread.join().unwrap(),
        )
    });

    let results = [forward_result, reverse_result];
    assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
    assert_eq!(
        results
            .iter()
            .filter(|result| **result == Err(TensorError::BackwardGraphFreed))
            .count(),
        1
    );
    assert_eq!(
        leaf.grad().unwrap().unwrap().item().unwrap().to_bits(),
        6.0_f32.to_bits()
    );
}

#[test]
fn multiply_backward_unbroadcasts_both_operands() {
    let left = Tensor::from_vec(vec![2.0, 3.0], [2, 1])
        .unwrap()
        .with_requires_grad(true);
    let right = Tensor::from_vec(vec![5.0, 7.0, 11.0], [1, 3])
        .unwrap()
        .with_requires_grad(true);

    left.mul(&right).unwrap().sum().backward().unwrap();

    assert_eq!(values(&left.grad().unwrap().unwrap()), [23.0, 23.0]);
    assert_eq!(values(&right.grad().unwrap().unwrap()), [5.0, 5.0, 5.0]);
}

#[test]
fn rank_zero_tensor_multiply_preserves_view_gradients_in_both_operand_orders() {
    for scalar_on_left in [true, false] {
        let mut input_values = vec![0.0; 6];
        input_values.extend([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let source = Tensor::from_vec(input_values, [2, 2, 3])
            .unwrap()
            .with_requires_grad(true);
        let view = source.index([1]).unwrap().transpose(0, 1).unwrap();
        let scalar = Tensor::from_vec(vec![2.0], [])
            .unwrap()
            .with_requires_grad(true);

        let output = if scalar_on_left {
            scalar.mul(&view).unwrap()
        } else {
            view.mul(&scalar).unwrap()
        };
        assert_eq!(output.shape(), [3, 2]);
        assert_eq!(output.stride(), [1, 3]);
        output.sum().backward().unwrap();

        assert_eq!(values(&scalar.grad().unwrap().unwrap()), [21.0]);
        assert_eq!(
            values(&source.grad().unwrap().unwrap()),
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        );
    }
}

#[test]
fn scalar_and_empty_reductions_produce_correct_leaf_gradients() {
    let scalar = Tensor::from_vec(vec![4.0], [])
        .unwrap()
        .with_requires_grad(true);
    scalar.mul_scalar(3.0).unwrap().backward().unwrap();
    assert_eq!(
        scalar.grad().unwrap().unwrap().item().unwrap().to_bits(),
        3.0_f32.to_bits()
    );

    let empty = Tensor::zeros([2, 0, 3]).unwrap().with_requires_grad(true);
    let output = empty.sum();
    assert!(output.item().unwrap().abs() < f32::EPSILON);
    output.backward().unwrap();
    let gradient = empty.grad().unwrap().unwrap();
    assert_eq!(gradient.shape(), [2, 0, 3]);
    assert!(values(&gradient).is_empty());
}

#[test]
fn relu_vjp_selects_upstream_for_positives_and_nans() {
    let input_bits = [
        0x3f80_0000,
        0xbf80_0000,
        0x0000_0000,
        0x8000_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc1_2345,
        0xffc5_4321,
        0x0000_0001,
        0x8000_0001,
        0x7f7f_ffff,
        0xff7f_ffff,
    ];
    let weight_bits = [
        0x8000_0000,
        0x8000_0000,
        0xff80_0000,
        0x7f80_0000,
        0x7f80_0000,
        0xff80_0000,
        0x3f00_0000,
        0xbf00_0000,
        0xc040_0000,
        0x4040_0000,
        0x4000_0000,
        0x7fc0_1234,
    ];
    let expected_gradient_bits = [
        0x8000_0000,
        0x0000_0000,
        0x0000_0000,
        0x0000_0000,
        0x7f80_0000,
        0x0000_0000,
        0x3f00_0000,
        0xbf00_0000,
        0xc040_0000,
        0x0000_0000,
        0x4000_0000,
        0x0000_0000,
    ];
    let leaf = Tensor::from_vec(input_bits.map(f32::from_bits).to_vec(), [input_bits.len()])
        .unwrap()
        .with_requires_grad(true);
    let weights = Tensor::from_vec(
        weight_bits.map(f32::from_bits).to_vec(),
        [weight_bits.len()],
    )
    .unwrap();
    let output = leaf.relu().unwrap();

    assert!(output.requires_grad());
    assert!(!output.is_leaf());
    assert!(!output.shares_storage_with(&leaf));
    let loss = output.mul(&weights).unwrap().sum();
    loss.backward().unwrap();

    assert!(
        leaf.grad()
            .unwrap()
            .unwrap()
            .logical_values()
            .map(f32::to_bits)
            .eq(expected_gradient_bits)
    );
    assert_eq!(loss.backward(), Err(TensorError::BackwardGraphFreed));
}

#[test]
fn relu_preserves_scalar_empty_offset_and_strided_autograd() {
    let scalar = Tensor::from_vec(vec![2.0], [])
        .unwrap()
        .with_requires_grad(true);
    let scalar_output = scalar.relu().unwrap();
    assert!(scalar_output.requires_grad());
    assert!(!scalar_output.is_leaf());
    assert!(scalar_output.shape().is_empty());
    assert!(scalar_output.stride().is_empty());
    scalar_output.backward().unwrap();
    assert_eq!(
        scalar.grad().unwrap().unwrap().item().unwrap().to_bits(),
        1.0_f32.to_bits()
    );

    let empty = Tensor::zeros([2, 0, 3]).unwrap().with_requires_grad(true);
    let empty_loss = empty.relu().unwrap().sum();
    assert!(empty_loss.requires_grad());
    empty_loss.backward().unwrap();
    let empty_gradient = empty.grad().unwrap().unwrap();
    assert_eq!(empty_gradient.shape(), [2, 0, 3]);
    assert_eq!(empty_gradient.stride(), [3, 3, 1]);
    assert!(values(&empty_gradient).is_empty());
    assert_eq!(empty_loss.backward(), Err(TensorError::BackwardGraphFreed));

    let mut storage = vec![9.0; 12];
    storage.extend([
        -1.0,
        2.0,
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7fc1_2345),
        3.0,
        -4.0,
        5.0,
        -6.0,
        7.0,
    ]);
    let source = Tensor::from_vec(storage, [2, 3, 4])
        .unwrap()
        .with_requires_grad(true);
    let offset = source.index([1]).unwrap();
    let offset_output = offset.relu().unwrap();
    assert!(offset_output.requires_grad());
    assert_eq!(offset.storage_offset(), 12);
    assert_eq!(offset_output.shape(), [3, 4]);
    assert_eq!(offset_output.stride(), [4, 1]);
    assert_eq!(offset_output.storage_offset(), 0);
    assert!(!offset_output.shares_storage_with(&offset));
    offset_output.sum().backward().unwrap();

    let strided = offset.transpose(0, 1).unwrap();
    let strided_output = strided.relu().unwrap();
    assert!(strided_output.requires_grad());
    assert_eq!(strided.storage_offset(), 12);
    assert_eq!(strided_output.shape(), [4, 3]);
    assert_eq!(strided_output.stride(), [1, 4]);
    assert_eq!(strided_output.storage_offset(), 0);
    assert!(!strided_output.shares_storage_with(&strided));
    let weights = Tensor::from_vec((1_u8..=12).map(f32::from).collect(), [4, 3]).unwrap();
    strided_output
        .mul(&weights)
        .unwrap()
        .sum()
        .backward()
        .unwrap();

    assert_eq!(
        values(&source.grad().unwrap().unwrap()),
        [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 3.0,
            0.0, 9.0, 12.0, 0.0, 7.0, 0.0, 13.0,
        ]
    );
}

#[test]
fn relu_obeys_detach_no_grad_and_freed_graph_boundaries() {
    let leaf = Tensor::from_vec(vec![-1.0, 2.0, 0.0, f32::NAN], [2, 2])
        .unwrap()
        .with_requires_grad(true);
    let detached_input = leaf.detach().unwrap().relu().unwrap();
    assert!(!detached_input.requires_grad());

    let tracked = leaf.relu().unwrap();
    let detached_output = tracked.detach().unwrap();
    assert!(!detached_output.requires_grad());
    assert!(detached_output.shares_storage_with(&tracked));

    {
        let _guard = no_grad();
        let output = leaf.transpose(0, 1).unwrap().relu().unwrap();
        assert!(!output.requires_grad());
        assert_eq!(output.shape(), [2, 2]);
        assert_eq!(output.stride(), [1, 2]);
        assert_eq!(output.storage_offset(), 0);
    }
    assert!(leaf.relu().unwrap().requires_grad());

    let no_grad_view = {
        let _guard = no_grad();
        leaf.transpose(0, 1).unwrap()
    };
    let boundary_loss = no_grad_view.relu().unwrap().sum();
    assert!(boundary_loss.requires_grad());
    boundary_loss.backward().unwrap();
    assert!(leaf.grad().unwrap().is_none());
    assert_eq!(
        boundary_loss.backward(),
        Err(TensorError::BackwardGraphFreed)
    );

    let tracked_loss = tracked.sum();
    tracked_loss.backward().unwrap();
    assert_eq!(values(&leaf.grad().unwrap().unwrap()), [0.0, 1.0, 0.0, 1.0]);
    assert_eq!(
        tracked_loss.backward(),
        Err(TensorError::BackwardGraphFreed)
    );
}

#[test]
fn sine_preserves_scalar_empty_and_strided_autograd_history() {
    let scalar = Tensor::from_vec(vec![1.5], [])
        .unwrap()
        .with_requires_grad(true);
    let scalar_output = scalar.sin().unwrap();
    assert!(scalar_output.requires_grad());
    assert!(scalar_output.shape().is_empty());
    assert!(scalar_output.stride().is_empty());
    assert_eq!(scalar_output.storage_offset(), 0);
    scalar_output.backward().unwrap();
    assert_eq!(
        scalar.grad().unwrap().unwrap().item().unwrap().to_bits(),
        1.5_f32.cos().to_bits()
    );

    let empty = Tensor::zeros([2, 0, 3]).unwrap().with_requires_grad(true);
    let empty_output = empty.sin().unwrap();
    assert!(empty_output.requires_grad());
    assert_eq!(empty_output.shape(), [2, 0, 3]);
    assert_eq!(empty_output.stride(), [3, 3, 1]);
    assert_eq!(empty_output.storage_offset(), 0);
    assert_eq!(empty_output.dtype(), empty.dtype());
    assert_eq!(empty_output.device(), empty.device());
    empty_output.sum().backward().unwrap();
    let empty_gradient = empty.grad().unwrap().unwrap();
    assert_eq!(empty_gradient.shape(), [2, 0, 3]);
    assert_eq!(empty_gradient.stride(), [3, 3, 1]);
    assert!(values(&empty_gradient).is_empty());

    let leaf = Tensor::from_vec(vec![-2.0, 0.0, 1.0, 2.0, 4.0, 6.0], [2, 3])
        .unwrap()
        .with_requires_grad(true);
    let view = leaf.transpose(0, 1).unwrap();
    let weights = Tensor::from_vec(vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0], [3, 2]).unwrap();
    let output = view.sin().unwrap();
    assert!(output.requires_grad());
    assert_eq!(output.shape(), [3, 2]);
    assert_eq!(output.stride(), [1, 3]);
    assert_eq!(output.storage_offset(), 0);
    assert_eq!(output.dtype(), view.dtype());
    assert_eq!(output.device(), view.device());
    output.mul(&weights).unwrap().sum().backward().unwrap();
    assert_eq!(
        values(&leaf.grad().unwrap().unwrap()),
        [
            (-2.0_f32).cos(),
            3.0 * 0.0_f32.cos(),
            5.0 * 1.0_f32.cos(),
            -2.0 * 2.0_f32.cos(),
            -4.0 * 4.0_f32.cos(),
            -6.0 * 6.0_f32.cos(),
        ]
    );
}

#[test]
fn sine_vjp_uses_saved_input_for_signed_zero_non_finites_and_nans() {
    let input_bits = [
        0x0000_0000,
        0x8000_0000,
        0x3f00_0000,
        0xbf00_0000,
        0x3f80_0000,
        0xc000_0000,
        0x4049_0fdb,
        0x5015_02f9,
        0x7f80_0000,
        0xff80_0000,
        0x7fc1_2345,
        0xffc5_4321,
    ];
    let weight_bits = [
        0x3f80_0000,
        0xbf80_0000,
        0x0000_0000,
        0x8000_0000,
        0x7f80_0000,
        0xff80_0000,
        0x3f00_0000,
        0xbf00_0000,
        0x0000_0000,
        0x7f80_0000,
        0x3f80_0000,
        0xbf80_0000,
    ];
    let input_values = input_bits.map(f32::from_bits);
    let weight_values = weight_bits.map(f32::from_bits);
    let expected_bits = input_values
        .iter()
        .zip(weight_values)
        .map(|(&input, upstream)| (upstream * input.cos()).to_bits())
        .collect::<Vec<_>>();
    let leaf = Tensor::from_vec(input_values.to_vec(), [input_bits.len()])
        .unwrap()
        .with_requires_grad(true);
    let weights = Tensor::from_vec(weight_values.to_vec(), [weight_bits.len()]).unwrap();
    let loss = leaf.sin().unwrap().mul(&weights).unwrap().sum();

    loss.backward().unwrap();
    assert!(
        leaf.grad()
            .unwrap()
            .unwrap()
            .logical_values()
            .map(f32::to_bits)
            .eq(expected_bits)
    );
    assert_eq!(loss.backward(), Err(TensorError::BackwardGraphFreed));
}

#[test]
fn sine_obeys_detach_no_grad_and_freed_graph_boundaries() {
    let leaf = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2])
        .unwrap()
        .with_requires_grad(true);
    let detached_input = leaf.detach().unwrap().sin().unwrap();
    assert!(!detached_input.requires_grad());

    let tracked = leaf.sin().unwrap();
    let detached_output = tracked.detach().unwrap();
    assert!(!detached_output.requires_grad());
    assert!(detached_output.shares_storage_with(&tracked));

    {
        let _guard = no_grad();
        let output = leaf.transpose(0, 1).unwrap().sin().unwrap();
        assert!(!output.requires_grad());
        assert_eq!(output.shape(), [2, 2]);
        assert_eq!(output.stride(), [1, 2]);
        assert_eq!(output.storage_offset(), 0);
    }
    assert!(leaf.sin().unwrap().requires_grad());

    let no_grad_view = {
        let _guard = no_grad();
        leaf.transpose(0, 1).unwrap()
    };
    let boundary_loss = no_grad_view.sin().unwrap().sum();
    assert!(boundary_loss.requires_grad());
    boundary_loss.backward().unwrap();
    assert!(leaf.grad().unwrap().is_none());
    assert_eq!(
        boundary_loss.backward(),
        Err(TensorError::BackwardGraphFreed)
    );

    let tracked_loss = tracked.sum();
    tracked_loss.backward().unwrap();
    assert_eq!(
        values(&leaf.grad().unwrap().unwrap()),
        [1.0_f32.cos(), 2.0_f32.cos(), 3.0_f32.cos(), 4.0_f32.cos()]
    );
    assert_eq!(
        tracked_loss.backward(),
        Err(TensorError::BackwardGraphFreed)
    );
}

#[test]
fn exponential_preserves_scalar_empty_offset_and_strided_autograd_history() {
    let scalar = Tensor::from_vec(vec![1.5], [])
        .unwrap()
        .with_requires_grad(true);
    let scalar_output = scalar.exp().unwrap();
    assert!(scalar_output.requires_grad());
    assert!(!scalar_output.is_leaf());
    assert!(scalar_output.shape().is_empty());
    assert!(scalar_output.stride().is_empty());
    assert_eq!(scalar_output.storage_offset(), 0);
    scalar_output.backward().unwrap();
    assert_eq!(
        scalar.grad().unwrap().unwrap().item().unwrap().to_bits(),
        1.5_f32.exp().to_bits()
    );

    let empty = Tensor::zeros([2, 0, 3]).unwrap().with_requires_grad(true);
    let empty_output = empty.exp().unwrap();
    assert!(empty_output.requires_grad());
    assert!(!empty_output.is_leaf());
    assert_eq!(empty_output.shape(), [2, 0, 3]);
    assert_eq!(empty_output.stride(), [3, 3, 1]);
    assert_eq!(empty_output.storage_offset(), 0);
    assert_eq!(empty_output.dtype(), empty.dtype());
    assert_eq!(empty_output.device(), empty.device());
    empty_output.sum().backward().unwrap();
    let empty_gradient = empty.grad().unwrap().unwrap();
    assert_eq!(empty_gradient.shape(), [2, 0, 3]);
    assert_eq!(empty_gradient.stride(), [3, 3, 1]);
    assert!(values(&empty_gradient).is_empty());

    let offset_leaf = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], [2, 3])
        .unwrap()
        .with_requires_grad(true);
    let offset = offset_leaf.index([1]).unwrap();
    let offset_weights = Tensor::from_vec(vec![1.0, 2.0, 3.0], [3]).unwrap();
    let offset_output = offset.exp().unwrap();
    assert!(offset_output.requires_grad());
    assert!(!offset_output.is_leaf());
    assert_eq!(offset.storage_offset(), 3);
    assert_eq!(offset_output.shape(), [3]);
    assert_eq!(offset_output.stride(), [1]);
    assert_eq!(offset_output.storage_offset(), 0);
    assert!(!offset_output.shares_storage_with(&offset));
    offset_output
        .mul(&offset_weights)
        .unwrap()
        .sum()
        .backward()
        .unwrap();
    assert_eq!(
        values(&offset_leaf.grad().unwrap().unwrap()),
        [
            0.0,
            0.0,
            0.0,
            1.0_f32.exp(),
            2.0 * 2.0_f32.exp(),
            3.0 * 3.0_f32.exp(),
        ]
    );

    let strided_leaf = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], [2, 3])
        .unwrap()
        .with_requires_grad(true);
    let strided = strided_leaf.transpose(0, 1).unwrap();
    let strided_weights = Tensor::from_vec(vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0], [3, 2]).unwrap();
    let strided_output = strided.exp().unwrap();
    assert!(strided_output.requires_grad());
    assert!(!strided_output.is_leaf());
    assert_eq!(strided_output.shape(), [3, 2]);
    assert_eq!(strided_output.stride(), [1, 3]);
    assert_eq!(strided_output.storage_offset(), 0);
    assert!(!strided_output.shares_storage_with(&strided));
    strided_output
        .mul(&strided_weights)
        .unwrap()
        .sum()
        .backward()
        .unwrap();
    assert_eq!(
        values(&strided_leaf.grad().unwrap().unwrap()),
        [
            (-2.0_f32).exp(),
            3.0 * (-1.0_f32).exp(),
            5.0,
            -2.0 * 1.0_f32.exp(),
            -4.0 * 2.0_f32.exp(),
            -6.0 * 3.0_f32.exp(),
        ]
    );
}

#[test]
fn exponential_vjp_matches_pytorch_overflow_subnormal_and_nonfinite_bits() {
    // input, upstream, forward result, gradient
    const CASES: [(u32, u32, u32, u32); 20] = [
        (0xff80_0000, 0x7f80_0000, 0x0000_0000, 0xffc0_0000),
        (0xc2d0_0000, 0xbf80_0000, 0x0000_0000, 0x8000_0000),
        (0xc2cf_0000, 0x3f00_0000, 0x0000_0001, 0x0000_0000),
        (0xc2ce_0000, 0xbf00_0000, 0x0000_0001, 0x8000_0000),
        (0xc2c8_0000, 0x0000_0001, 0x0000_001b, 0x0000_0000),
        (0xc2b0_0000, 0xff80_0000, 0x0041_edc4, 0xff80_0000),
        (0xbf80_0000, 0x0000_0000, 0x3ebc_5ab2, 0x0000_0000),
        (0x8000_0000, 0x8000_0000, 0x3f80_0000, 0x8000_0000),
        (0x0000_0000, 0x3f80_0000, 0x3f80_0000, 0x3f80_0000),
        (0x3f80_0000, 0xbf80_0000, 0x402d_f854, 0xc02d_f854),
        (0x4120_0000, 0x0000_0001, 0x46ac_14ee, 0x0000_560a),
        (0x42a0_0000, 0x3e80_0000, 0x792a_bbce, 0x782a_bbce),
        (0x42b0_0000, 0x4000_0000, 0x7ef8_82b7, 0x7f78_82b7),
        (0x42b1_8000, 0xbf80_0000, 0x7f80_0000, 0xff80_0000),
        (0x42b2_0000, 0x3f80_0000, 0x7f80_0000, 0x7f80_0000),
        (0x7f80_0000, 0x0000_0000, 0x7f80_0000, 0xffc0_0000),
        (0x7f81_2345, 0x3f80_0000, 0x7fc1_2345, 0x7fc1_2345),
        (0xff81_2345, 0xbf80_0000, 0xffc1_2345, 0xffc1_2345),
        (0x7fc1_2345, 0x7fc0_1234, 0x7fc1_2345, 0x7fc0_1234),
        (0xffc5_4321, 0xffc0_5678, 0xffc5_4321, 0xffc0_5678),
    ];
    let leaf = Tensor::from_vec(
        CASES
            .iter()
            .map(|&(input, _, _, _)| f32::from_bits(input))
            .collect(),
        [CASES.len()],
    )
    .unwrap()
    .with_requires_grad(true);
    let weights = Tensor::from_vec(
        CASES
            .iter()
            .map(|&(_, upstream, _, _)| f32::from_bits(upstream))
            .collect(),
        [CASES.len()],
    )
    .unwrap();
    let output = leaf.exp().unwrap();
    for (actual, (_, _, expected_bits, _)) in output.logical_values().zip(CASES) {
        let expected = f32::from_bits(expected_bits);
        if expected.is_nan() {
            assert!(actual.is_nan());
        } else {
            assert_eq!(actual.to_bits(), expected_bits);
        }
    }
    let loss = output.mul(&weights).unwrap().sum();

    loss.backward().unwrap();
    for (actual, (_, _, _, expected_bits)) in
        leaf.grad().unwrap().unwrap().logical_values().zip(CASES)
    {
        let expected = f32::from_bits(expected_bits);
        if expected.is_nan() {
            assert!(actual.is_nan());
        } else {
            assert_eq!(actual.to_bits(), expected_bits);
        }
    }
    assert_eq!(loss.backward(), Err(TensorError::BackwardGraphFreed));
}

#[test]
fn exponential_composes_accumulates_and_obeys_detach_and_no_grad() {
    let accumulated = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 4.0], [2, 2])
        .unwrap()
        .with_requires_grad(true);
    accumulated.exp().unwrap().sum().backward().unwrap();
    accumulated.exp().unwrap().sum().backward().unwrap();
    assert_eq!(
        values(&accumulated.grad().unwrap().unwrap()),
        [
            (-1.0_f32).exp() * 2.0,
            2.0,
            1.0_f32.exp() * 2.0,
            4.0_f32.exp() * 2.0,
        ]
    );

    let composed = Tensor::from_vec(vec![-1.0, 0.5, 2.0], [3])
        .unwrap()
        .with_requires_grad(true);
    composed
        .sin()
        .unwrap()
        .exp()
        .unwrap()
        .sum()
        .backward()
        .unwrap();
    assert_eq!(
        values(&composed.grad().unwrap().unwrap()),
        [
            (-1.0_f32).sin().exp() * (-1.0_f32).cos(),
            0.5_f32.sin().exp() * 0.5_f32.cos(),
            2.0_f32.sin().exp() * 2.0_f32.cos(),
        ]
    );

    assert!(!accumulated.detach().unwrap().exp().unwrap().requires_grad());
    {
        let _guard = no_grad();
        let output = accumulated.transpose(0, 1).unwrap().exp().unwrap();
        assert!(!output.requires_grad());
        assert!(output.is_leaf());
        assert_eq!(output.shape(), [2, 2]);
        assert_eq!(output.stride(), [1, 2]);
        assert_eq!(output.storage_offset(), 0);
    }
    assert!(accumulated.exp().unwrap().requires_grad());
}

#[test]
fn sqrt_preserves_scalar_empty_offset_and_strided_autograd_history() {
    let scalar = Tensor::from_vec(vec![4.0], [])
        .unwrap()
        .with_requires_grad(true);
    let scalar_output = scalar.sqrt().unwrap();
    assert!(scalar_output.requires_grad());
    assert!(!scalar_output.is_leaf());
    assert!(scalar_output.shape().is_empty());
    assert!(scalar_output.stride().is_empty());
    assert_eq!(scalar_output.storage_offset(), 0);
    scalar_output.backward().unwrap();
    assert_eq!(
        scalar.grad().unwrap().unwrap().item().unwrap().to_bits(),
        0.25_f32.to_bits()
    );

    let empty = Tensor::zeros([2, 0, 3]).unwrap().with_requires_grad(true);
    let empty_output = empty.sqrt().unwrap();
    assert!(empty_output.requires_grad());
    assert_eq!(empty_output.shape(), [2, 0, 3]);
    assert_eq!(empty_output.stride(), [3, 3, 1]);
    assert_eq!(empty_output.storage_offset(), 0);
    assert_eq!(empty_output.dtype(), empty.dtype());
    assert_eq!(empty_output.device(), empty.device());
    empty_output.sum().backward().unwrap();
    let empty_gradient = empty.grad().unwrap().unwrap();
    assert_eq!(empty_gradient.shape(), [2, 0, 3]);
    assert_eq!(empty_gradient.stride(), [3, 3, 1]);
    assert!(values(&empty_gradient).is_empty());

    let source = Tensor::from_vec((1_u8..=24).map(f32::from).collect(), [2, 3, 4])
        .unwrap()
        .with_requires_grad(true);
    let offset = source.index([1]).unwrap();
    let offset_output = offset.sqrt().unwrap();
    assert!(offset_output.requires_grad());
    assert_eq!(offset.storage_offset(), 12);
    assert_eq!(offset_output.shape(), [3, 4]);
    assert_eq!(offset_output.stride(), [4, 1]);
    assert_eq!(offset_output.storage_offset(), 0);
    assert!(!offset_output.shares_storage_with(&offset));
    offset_output.sum().backward().unwrap();
    let expected = (1_u8..=24)
        .map(|value| {
            if value <= 12 {
                0.0
            } else {
                1.0 / (2.0 * f32::from(value).sqrt())
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(values(&source.grad().unwrap().unwrap()), expected);

    let strided_source = Tensor::from_vec((1_u8..=24).map(f32::from).collect(), [2, 3, 4])
        .unwrap()
        .with_requires_grad(true);
    let strided = strided_source.index([1]).unwrap().transpose(0, 1).unwrap();
    let weights = Tensor::from_vec((1_u8..=12).map(f32::from).collect(), [4, 3]).unwrap();
    let strided_output = strided.sqrt().unwrap();
    assert!(strided_output.requires_grad());
    assert_eq!(strided.storage_offset(), 12);
    assert_eq!(strided_output.shape(), [4, 3]);
    assert_eq!(strided_output.stride(), [1, 4]);
    assert_eq!(strided_output.storage_offset(), 0);
    assert!(!strided_output.shares_storage_with(&strided));
    strided_output
        .mul(&weights)
        .unwrap()
        .sum()
        .backward()
        .unwrap();
    let expected = (1_u8..=24)
        .map(|value| {
            if value <= 12 {
                return 0.0;
            }
            let index = usize::from(value - 13);
            let row = index / 4;
            let column = index % 4;
            let upstream = f32::from(u8::try_from(column * 3 + row + 1).unwrap());
            upstream / (2.0 * f32::from(value).sqrt())
        })
        .collect::<Vec<_>>();
    assert_eq!(values(&strided_source.grad().unwrap().unwrap()), expected);
}

#[test]
fn sqrt_vjp_matches_pytorch_for_signed_zero_non_finites_and_nans() {
    let input_bits = [
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x8000_0001,
        0x0080_0000,
        0x8080_0000,
        0x3e80_0000,
        0x3f80_0000,
        0x4000_0000,
        0x4080_0000,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x7f80_0000,
        0xff80_0000,
        0x7f81_2345,
        0xff81_2345,
        0x7fc1_2345,
        0xffc5_4321,
    ];
    let weight_bits = [
        0x3f80_0000,
        0xbf80_0000,
        0x0000_0000,
        0x8000_0000,
        0x7f80_0000,
        0xff80_0000,
        0x3f00_0000,
        0xbf00_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x7fc0_1234,
        0xffc0_5678,
    ];
    let expected_gradient_bits = [
        0x7f80_0000,
        0x7f80_0000,
        0x0000_0000,
        0x7fc0_0000,
        0x7f80_0000,
        0x7fc0_0000,
        0x3f00_0000,
        0xbe80_0000,
        0x3eb5_04f3,
        0xbe80_0000,
        0x1f00_0001,
        0x7fc0_0000,
        0x0000_0000,
        0x7fc0_0000,
        0x7fc1_2345,
        0xffc1_2345,
        0x7fc0_1234,
        0xffc0_5678,
    ];
    let leaf = Tensor::from_vec(input_bits.map(f32::from_bits).to_vec(), [input_bits.len()])
        .unwrap()
        .with_requires_grad(true);
    let weights = Tensor::from_vec(
        weight_bits.map(f32::from_bits).to_vec(),
        [weight_bits.len()],
    )
    .unwrap();
    let loss = leaf.sqrt().unwrap().mul(&weights).unwrap().sum();

    loss.backward().unwrap();
    assert!(
        leaf.grad()
            .unwrap()
            .unwrap()
            .logical_values()
            .map(f32::to_bits)
            .eq(expected_gradient_bits)
    );
    assert_eq!(loss.backward(), Err(TensorError::BackwardGraphFreed));
}

#[test]
fn sqrt_accumulates_across_graphs_and_obeys_detach_and_no_grad() {
    let leaf = Tensor::from_vec(vec![1.0, 4.0, 9.0], [3])
        .unwrap()
        .with_requires_grad(true);
    leaf.sqrt().unwrap().sum().backward().unwrap();
    leaf.sqrt().unwrap().sum().backward().unwrap();
    assert_eq!(
        values(&leaf.grad().unwrap().unwrap()),
        [1.0, 0.5, 1.0 / 3.0]
    );

    assert!(!leaf.detach().unwrap().sqrt().unwrap().requires_grad());
    {
        let _guard = no_grad();
        let output = leaf.transpose(0, 0).unwrap().sqrt().unwrap();
        assert!(!output.requires_grad());
        assert_eq!(output.shape(), [3]);
        assert_eq!(output.stride(), [1]);
    }
    assert!(leaf.sqrt().unwrap().requires_grad());
}

#[test]
fn saved_input_unary_nodes_compose_and_release_their_saved_values() {
    let leaf = Tensor::from_vec(vec![-1.0, 0.5, 2.0, 4.0], [4])
        .unwrap()
        .with_requires_grad(true);
    let loss = leaf.sin().unwrap().relu().unwrap().sum();

    loss.backward().unwrap();
    assert_eq!(
        values(&leaf.grad().unwrap().unwrap()),
        [0.0, 0.5_f32.cos(), 2.0_f32.cos(), 0.0]
    );
    assert_eq!(loss.backward(), Err(TensorError::BackwardGraphFreed));
}

#[test]
fn scalar_addition_records_reusable_identity_gradients() {
    let leaf = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2])
        .unwrap()
        .with_requires_grad(true);
    let view = leaf.transpose(0, 1).unwrap();
    let weights = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]).unwrap();
    let output = view.add_scalar(5.0).unwrap().mul(&weights).unwrap().sum();

    assert!(output.requires_grad());
    output.backward().unwrap();
    assert_eq!(values(&leaf.grad().unwrap().unwrap()), [1.0, 3.0, 2.0, 4.0]);

    let repeated_leaf = Tensor::from_vec(vec![2.0, 3.0], [2])
        .unwrap()
        .with_requires_grad(true);
    let repeated = repeated_leaf.add_scalar(1.0).unwrap().sum();
    repeated.backward().unwrap();
    repeated.backward().unwrap();
    assert_eq!(values(&repeated_leaf.grad().unwrap().unwrap()), [2.0, 2.0]);

    let empty = Tensor::zeros([2, 0, 3]).unwrap().with_requires_grad(true);
    let empty_output = empty.add_scalar(7.0).unwrap();
    assert!(empty_output.requires_grad());
    empty_output.sum().backward().unwrap();
    let empty_gradient = empty.grad().unwrap().unwrap();
    assert_eq!(empty_gradient.shape(), [2, 0, 3]);
    assert!(values(&empty_gradient).is_empty());
}

#[test]
fn real_scalar_subtraction_records_reusable_signed_gradients() {
    let forward_leaf = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
        .unwrap()
        .with_requires_grad(true);
    let forward_view = forward_leaf.transpose(0, 1).unwrap();
    let weights = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2]).unwrap();
    let forward = forward_view.sub_scalar(2.0).unwrap();
    assert!(forward.requires_grad());
    assert_eq!(forward.stride(), [1, 3]);
    assert_eq!(values(&forward), [-1.0, 2.0, 0.0, 3.0, 1.0, 4.0]);
    forward.mul(&weights).unwrap().sum().backward().unwrap();
    assert_eq!(
        values(&forward_leaf.grad().unwrap().unwrap()),
        [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
    );

    let reflected_leaf = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
        .unwrap()
        .with_requires_grad(true);
    let reflected = reflected_leaf
        .transpose(0, 1)
        .unwrap()
        .scalar_sub(10.0)
        .unwrap();
    assert!(reflected.requires_grad());
    assert_eq!(reflected.stride(), [1, 3]);
    assert_eq!(values(&reflected), [9.0, 6.0, 8.0, 5.0, 7.0, 4.0]);
    reflected.mul(&weights).unwrap().sum().backward().unwrap();
    assert_eq!(
        values(&reflected_leaf.grad().unwrap().unwrap()),
        [-1.0, -3.0, -5.0, -2.0, -4.0, -6.0]
    );

    for reflected in [false, true] {
        let empty = Tensor::zeros([2, 0, 3]).unwrap().with_requires_grad(true);
        let empty_output = if reflected {
            empty.scalar_sub(7.0).unwrap()
        } else {
            empty.sub_scalar(7.0).unwrap()
        };
        assert!(empty_output.requires_grad());
        assert_eq!(empty_output.stride(), [3, 3, 1]);
        empty_output.sum().backward().unwrap();
        let gradient = empty.grad().unwrap().unwrap();
        assert_eq!(gradient.shape(), [2, 0, 3]);
        assert!(values(&gradient).is_empty());
    }

    let forward_repeated = Tensor::from_vec(vec![2.0, 3.0], [2])
        .unwrap()
        .with_requires_grad(true);
    let forward_loss = forward_repeated.sub_scalar(1.0).unwrap().sum();
    forward_loss.backward().unwrap();
    forward_loss.backward().unwrap();
    assert_eq!(
        values(&forward_repeated.grad().unwrap().unwrap()),
        [2.0, 2.0]
    );

    let reflected_repeated = Tensor::from_vec(vec![2.0, 3.0], [2])
        .unwrap()
        .with_requires_grad(true);
    let reflected_loss = reflected_repeated.scalar_sub(1.0).unwrap().sum();
    reflected_loss.backward().unwrap();
    reflected_loss.backward().unwrap();
    assert_eq!(
        values(&reflected_repeated.grad().unwrap().unwrap()),
        [-2.0, -2.0]
    );
}

#[test]
fn detach_and_nested_no_grad_are_graph_boundaries() {
    let x = Tensor::from_vec(vec![2.0], [])
        .unwrap()
        .with_requires_grad(true);
    let detached = x.detach().unwrap();
    assert!(!detached.requires_grad());
    assert!(detached.shares_storage_with(&x));
    assert_eq!(
        detached.mul(&detached).unwrap().backward(),
        Err(TensorError::DoesNotRequireGrad)
    );

    {
        let _outer = no_grad();
        assert!(!x.mul(&x).unwrap().requires_grad());
        {
            let _inner = no_grad();
            assert!(!x.sum().requires_grad());
        }
        assert!(!x.mul_scalar(2.0).unwrap().requires_grad());
        assert!(!x.add_scalar(2.0).unwrap().requires_grad());
        assert!(!x.sub_scalar(2.0).unwrap().requires_grad());
        assert!(!x.scalar_sub(2.0).unwrap().requires_grad());
    }
    assert!(x.mul(&x).unwrap().requires_grad());

    assert!(!detached.add_scalar(2.0).unwrap().requires_grad());
    assert!(!detached.sub_scalar(2.0).unwrap().requires_grad());
    assert!(!detached.scalar_sub(2.0).unwrap().requires_grad());
}

#[test]
fn backward_errors_are_stable_and_saved_graphs_live_until_consumed() {
    let plain = Tensor::from_vec(vec![1.0], []).unwrap();
    assert_eq!(plain.backward(), Err(TensorError::DoesNotRequireGrad));
    let plain_vector = Tensor::from_vec(vec![1.0, 2.0], [2]).unwrap();
    assert_eq!(
        plain_vector.backward(),
        Err(TensorError::DoesNotRequireGrad)
    );

    let vector = Tensor::from_vec(vec![1.0, 2.0], [2])
        .unwrap()
        .with_requires_grad(true);
    assert_eq!(
        vector.backward(),
        Err(TensorError::BackwardRequiresScalar { elements: 2 })
    );

    let leaf = Tensor::from_vec(vec![2.0, 3.0], [2])
        .unwrap()
        .with_requires_grad(true);
    let output = {
        let intermediate = leaf.mul(&leaf).unwrap();
        intermediate.sum()
    };
    output.backward().unwrap();
    assert_eq!(values(&leaf.grad().unwrap().unwrap()), [4.0, 6.0]);
    assert_eq!(output.backward(), Err(TensorError::BackwardGraphFreed));
}

#[test]
fn one_element_nonscalar_leaf_can_seed_implicit_backward_repeatedly() {
    let leaf = Tensor::from_vec(vec![9.0], [1])
        .unwrap()
        .with_requires_grad(true);
    leaf.backward().unwrap();
    leaf.backward().unwrap();
    assert_eq!(values(&leaf.grad().unwrap().unwrap()), [2.0]);
}

#[test]
fn metadata_only_graphs_support_repeated_backward() {
    let summed_leaf = Tensor::from_vec(vec![1.0, 2.0], [2])
        .unwrap()
        .with_requires_grad(true);
    let sum = summed_leaf.sum();
    sum.backward().unwrap();
    sum.backward().unwrap();
    assert_eq!(values(&summed_leaf.grad().unwrap().unwrap()), [2.0, 2.0]);

    let transformed_leaf = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2])
        .unwrap()
        .with_requires_grad(true);
    let transformed_sum = transformed_leaf
        .transpose(0, 1)
        .unwrap()
        .try_clone()
        .unwrap()
        .sum();
    transformed_sum.backward().unwrap();
    transformed_sum.backward().unwrap();
    assert_eq!(
        values(&transformed_leaf.grad().unwrap().unwrap()),
        [2.0, 2.0, 2.0, 2.0]
    );
}

#[test]
fn no_grad_guards_remain_disabled_until_every_guard_is_dropped() {
    let leaf = Tensor::from_vec(vec![2.0], [])
        .unwrap()
        .with_requires_grad(true);
    let outer = no_grad();
    let inner = no_grad();

    drop(outer);
    assert!(!leaf.mul_scalar(2.0).unwrap().requires_grad());

    drop(inner);
    assert!(leaf.mul_scalar(2.0).unwrap().requires_grad());
}

#[test]
fn grad_enabled_state_is_nested_exception_safe_and_thread_local() {
    assert!(is_grad_enabled());
    {
        let _outer = no_grad();
        assert!(!is_grad_enabled());
        {
            let _inner = no_grad();
            assert!(!is_grad_enabled());
        }
        assert!(!is_grad_enabled());
    }
    assert!(is_grad_enabled());

    let unwind = std::panic::catch_unwind(|| {
        let _guard = no_grad();
        assert!(!is_grad_enabled());
        panic!("restore grad mode");
    });
    assert!(unwind.is_err());
    assert!(is_grad_enabled());

    let guard = no_grad();
    assert!(!is_grad_enabled());
    thread::spawn(|| {
        assert!(is_grad_enabled());
        {
            let _guard = no_grad();
            assert!(!is_grad_enabled());
        }
        assert!(is_grad_enabled());
    })
    .join()
    .unwrap();
    assert!(!is_grad_enabled());
    drop(guard);
    assert!(is_grad_enabled());
}

#[test]
fn deep_graph_backward_uses_an_iterative_topology_walk() {
    let leaf = Tensor::from_vec(vec![3.0], [])
        .unwrap()
        .with_requires_grad(true);
    let mut output = leaf.mul_scalar(1.0).unwrap();
    for _ in 0..20_000 {
        output = output.mul_scalar(1.0).unwrap();
    }

    output.backward().unwrap();
    assert_eq!(
        leaf.grad().unwrap().unwrap().item().unwrap().to_bits(),
        1.0_f32.to_bits()
    );
}

#[test]
fn unconsumed_deep_graph_drop_and_detach_are_stack_safe() {
    let leaf = Tensor::from_vec(vec![3.0], [])
        .unwrap()
        .with_requires_grad(true);

    let mut output = leaf.mul_scalar(1.0).unwrap();
    for _ in 0..100_000 {
        output = output.mul_scalar(1.0).unwrap();
    }
    let detached = output.detach().unwrap();
    drop(output);
    assert!(!detached.requires_grad());
    assert_eq!(detached.item().unwrap().to_bits(), 3.0_f32.to_bits());

    let mut unconsumed = leaf.mul_scalar(1.0).unwrap();
    for _ in 0..100_000 {
        unconsumed = unconsumed.mul_scalar(1.0).unwrap();
    }
    drop(unconsumed);
}

#[test]
fn transformations_record_inverse_gradient_mappings() {
    let leaf = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1, 2, 3])
        .unwrap()
        .with_requires_grad(true);
    let permuted = leaf.permute_axes([2, 0, 1]).unwrap();
    assert!(permuted.requires_grad());
    let packed = permuted.try_contiguous(MemoryFormat::Contiguous).unwrap();
    assert!(packed.requires_grad());
    let squeezed = packed.squeeze_dim(1).unwrap();
    assert!(squeezed.requires_grad());
    let indexed = squeezed.index([1]).unwrap();
    assert!(indexed.requires_grad());
    let reshaped = indexed.reshape([2, 1]).unwrap();
    assert!(reshaped.requires_grad());
    let cloned = reshaped.try_clone().unwrap();
    assert!(cloned.requires_grad());

    cloned.mul(&cloned).unwrap().sum().backward().unwrap();
    assert_eq!(
        values(&leaf.grad().unwrap().unwrap()),
        [0.0, 4.0, 0.0, 0.0, 10.0, 0.0]
    );

    let reshape_leaf = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
        .unwrap()
        .with_requires_grad(true);
    let reshaped_copy = reshape_leaf.transpose(0, 1).unwrap().reshape([6]).unwrap();
    let weights = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], [6]).unwrap();
    reshaped_copy
        .mul(&weights)
        .unwrap()
        .sum()
        .backward()
        .unwrap();
    assert_eq!(
        values(&reshape_leaf.grad().unwrap().unwrap()),
        [10.0, 30.0, 50.0, 20.0, 40.0, 60.0]
    );
}

#[test]
fn channels_last_clone_records_identity_gradients_and_obeys_no_grad() {
    let leaf = Tensor::from_vec((1_u8..=48).map(f32::from).collect(), [2, 3, 2, 4])
        .unwrap()
        .with_requires_grad(true);
    let source = leaf.transpose(0, 3).unwrap();
    let cloned = source
        .try_clone_with_memory_format(MemoryFormat::ChannelsLast)
        .unwrap();
    assert_eq!(cloned.stride(), [12, 1, 6, 3]);
    assert!(cloned.requires_grad());
    assert!(!cloned.is_leaf());
    assert!(!cloned.shares_storage_with(&source));
    drop(source);

    cloned.sum().backward().unwrap();
    assert_eq!(values(&leaf.grad().unwrap().unwrap()), vec![1.0; 48]);

    let no_grad_source = leaf.transpose(0, 3).unwrap();
    let _guard = no_grad();
    let no_grad_clone = no_grad_source
        .try_clone_with_memory_format(MemoryFormat::ChannelsLast)
        .unwrap();
    assert_eq!(no_grad_clone.stride(), [12, 1, 6, 3]);
    assert!(!no_grad_clone.requires_grad());
    assert!(no_grad_clone.is_leaf());
    assert!(!no_grad_clone.shares_storage_with(&no_grad_source));
}

#[test]
fn channels_last_3d_clone_records_identity_gradients_and_obeys_no_grad() {
    let leaf = Tensor::from_vec((1_u16..=240).map(f32::from).collect(), [2, 3, 2, 4, 5])
        .unwrap()
        .with_requires_grad(true);
    let source = leaf.transpose(0, 4).unwrap();
    let cloned = source
        .try_clone_with_memory_format(MemoryFormat::ChannelsLast3d)
        .unwrap();
    assert_eq!(cloned.stride(), [48, 1, 24, 6, 3]);
    assert!(cloned.requires_grad());
    assert!(!cloned.is_leaf());
    assert!(!cloned.shares_storage_with(&source));
    drop(source);

    cloned.sum().backward().unwrap();
    assert_eq!(values(&leaf.grad().unwrap().unwrap()), vec![1.0; 240]);

    let no_grad_source = leaf.transpose(0, 4).unwrap();
    let _guard = no_grad();
    let no_grad_clone = no_grad_source
        .try_clone_with_memory_format(MemoryFormat::ChannelsLast3d)
        .unwrap();
    assert_eq!(no_grad_clone.stride(), [48, 1, 24, 6, 3]);
    assert!(!no_grad_clone.requires_grad());
    assert!(no_grad_clone.is_leaf());
    assert!(!no_grad_clone.shares_storage_with(&no_grad_source));
}

#[test]
fn ravel_records_view_and_copy_gradients_and_obeys_no_grad() {
    let leaf = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
        .unwrap()
        .with_requires_grad(true);
    let raveled = leaf.transpose(0, 1).unwrap().ravel().unwrap();
    assert_eq!(raveled.shape(), [6]);
    assert_eq!(raveled.stride(), [1]);
    assert!(!raveled.shares_storage_with(&leaf));
    assert!(raveled.requires_grad());
    assert!(!raveled.is_leaf());

    let weights = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], [6]).unwrap();
    raveled.mul(&weights).unwrap().sum().backward().unwrap();
    assert_eq!(
        values(&leaf.grad().unwrap().unwrap()),
        [10.0, 30.0, 50.0, 20.0, 40.0, 60.0]
    );

    let scalar = Tensor::from_vec(vec![2.0], [])
        .unwrap()
        .with_requires_grad(true);
    scalar
        .ravel()
        .unwrap()
        .mul_scalar(7.0)
        .unwrap()
        .sum()
        .backward()
        .unwrap();
    assert_eq!(
        scalar.grad().unwrap().unwrap().item().unwrap().to_bits(),
        7.0_f32.to_bits()
    );

    let no_grad_leaf = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2])
        .unwrap()
        .with_requires_grad(true);
    let no_grad_transposed = no_grad_leaf.transpose(0, 1).unwrap();
    let _guard = no_grad();
    let alias = no_grad_leaf.ravel().unwrap();
    assert!(alias.requires_grad());
    assert!(alias.is_leaf());
    assert!(alias.shares_storage_with(&no_grad_leaf));

    let copy = no_grad_transposed.ravel().unwrap();
    assert!(!copy.requires_grad());
    assert!(copy.is_leaf());
    assert!(!copy.shares_storage_with(&no_grad_leaf));
}

#[test]
fn no_grad_views_preserve_requires_grad_without_recording_history() {
    let source = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2])
        .unwrap()
        .with_requires_grad(true);

    let _guard = no_grad();
    let transposed = source.transpose(0, 1).unwrap();
    assert!(transposed.requires_grad());
    assert!(source.reshape([4]).unwrap().requires_grad());
    assert!(source.squeeze().unwrap().requires_grad());
    assert!(source.index([0]).unwrap().requires_grad());
    assert_eq!(
        transposed.backward(),
        Err(TensorError::BackwardRequiresScalar { elements: 4 })
    );

    assert!(!source.try_clone().unwrap().requires_grad());
    assert!(
        source
            .try_contiguous(MemoryFormat::Contiguous)
            .unwrap()
            .requires_grad()
    );
    assert!(
        !transposed
            .try_contiguous(MemoryFormat::Contiguous)
            .unwrap()
            .requires_grad()
    );
}

#[test]
fn leaf_status_reflects_recorded_autograd_history() {
    let ordinary = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]).unwrap();
    assert!(ordinary.is_leaf());
    assert!(ordinary.mul_scalar(2.0).unwrap().is_leaf());
    assert!(ordinary.transpose(0, 1).unwrap().is_leaf());

    let leaf = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2])
        .unwrap()
        .with_requires_grad(true);
    assert!(leaf.is_leaf());

    let operation = leaf.mul_scalar(2.0).unwrap();
    let view = leaf.transpose(0, 1).unwrap();
    assert!(!operation.is_leaf());
    assert!(!view.is_leaf());
    assert!(operation.detach().unwrap().is_leaf());

    let no_grad_views = {
        let _guard = no_grad();
        let no_grad_operation = leaf.mul_scalar(2.0).unwrap();
        assert!(no_grad_operation.is_leaf());
        [
            leaf.transpose(0, 1).unwrap(),
            operation.transpose(0, 1).unwrap(),
        ]
    };
    for no_grad_view in no_grad_views {
        assert!(no_grad_view.requires_grad());
        assert!(no_grad_view.is_leaf());
        assert!(!no_grad_view.mul_scalar(2.0).unwrap().is_leaf());
    }
}

#[test]
fn retains_grad_is_false_for_every_supported_autograd_state() {
    let ordinary = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]).unwrap();
    let ordinary_operation = ordinary.mul_scalar(2.0).unwrap();
    let ordinary_view = ordinary.transpose(0, 1).unwrap();

    let leaf = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2])
        .unwrap()
        .with_requires_grad(true);
    let operation = leaf.mul_scalar(2.0).unwrap();
    let view = leaf.transpose(0, 1).unwrap();
    let detached_operation = operation.detach().unwrap();
    let detached_view = view.detach().unwrap();

    let (no_grad_operation, no_grad_leaf_view, no_grad_non_leaf_view) = {
        let _guard = no_grad();
        (
            leaf.mul_scalar(3.0).unwrap(),
            leaf.transpose(0, 1).unwrap(),
            operation.transpose(0, 1).unwrap(),
        )
    };
    let recorded_after_no_grad = no_grad_leaf_view.mul_scalar(4.0).unwrap();

    operation.sum().backward().unwrap();
    let live_gradient = leaf.grad().unwrap().unwrap();

    for tensor in [
        &ordinary,
        &ordinary_operation,
        &ordinary_view,
        &leaf,
        &operation,
        &view,
        &detached_operation,
        &detached_view,
        &no_grad_operation,
        &no_grad_leaf_view,
        &no_grad_non_leaf_view,
        &recorded_after_no_grad,
        &live_gradient,
    ] {
        assert!(!tensor.retains_grad());
    }
}

#[test]
fn output_number_is_zero_for_every_supported_single_output_autograd_state() {
    let ordinary = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]).unwrap();
    let ordinary_operation = ordinary.mul_scalar(2.0).unwrap();
    let ordinary_view = ordinary.transpose(0, 1).unwrap();

    let leaf = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2])
        .unwrap()
        .with_requires_grad(true);
    let operation = leaf.mul_scalar(2.0).unwrap();
    let view = operation.transpose(0, 1).unwrap();
    let detached_operation = operation.detach().unwrap();
    let detached_view = view.detach().unwrap();
    let empty = Tensor::zeros([2, 0, 3]).unwrap().with_requires_grad(true);
    let empty_operation = empty.mul_scalar(2.0).unwrap();

    let (no_grad_operation, no_grad_leaf_view, no_grad_non_leaf_view) = {
        let _guard = no_grad();
        (
            leaf.mul_scalar(3.0).unwrap(),
            leaf.transpose(0, 1).unwrap(),
            operation.transpose(0, 1).unwrap(),
        )
    };

    operation.sum().backward().unwrap();
    let live_gradient = leaf.grad().unwrap().unwrap();

    for tensor in [
        &ordinary,
        &ordinary_operation,
        &ordinary_view,
        &leaf,
        &operation,
        &view,
        &detached_operation,
        &detached_view,
        &empty,
        &empty_operation,
        &no_grad_operation,
        &no_grad_leaf_view,
        &no_grad_non_leaf_view,
        &live_gradient,
    ] {
        assert_eq!(tensor.output_nr(), 0);
    }
}

#[test]
fn multiply_backward_preserves_first_negative_zero_contribution() {
    let left = Tensor::from_vec(vec![2.0], [1])
        .unwrap()
        .with_requires_grad(true);
    let right = Tensor::from_vec(vec![-0.0], [1]).unwrap();

    left.mul(&right).unwrap().sum().backward().unwrap();

    assert_eq!(
        values(&left.grad().unwrap().unwrap())[0].to_bits(),
        (-0.0_f32).to_bits()
    );

    let broadcast_left = Tensor::from_vec(vec![2.0], [])
        .unwrap()
        .with_requires_grad(true);
    let broadcast_right = Tensor::from_vec(vec![-0.0, -0.0], [2]).unwrap();
    broadcast_left
        .mul(&broadcast_right)
        .unwrap()
        .sum()
        .backward()
        .unwrap();
    assert_eq!(
        broadcast_left
            .grad()
            .unwrap()
            .unwrap()
            .item()
            .unwrap()
            .to_bits(),
        0.0_f32.to_bits()
    );
}
