use pytorch_rs::{MemoryFormat, Tensor, TensorError, no_grad};
use std::{
    sync::{Arc, Barrier},
    thread,
};

fn values(tensor: &Tensor) -> Vec<f32> {
    tensor.try_to_vec().unwrap()
}

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
fn tensor_by_scalar_division_records_float32_vjps_for_special_values_views_and_empties() {
    let upstream = [
        0x46d7_5128, // Distinguishes division from reciprocal multiplication.
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
    ]
    .map(f32::from_bits);
    let scalar_bits = [
        0x383a_7098,
        0x4000_0000,
        0xc000_0000,
        0x0000_0000,
        0x8000_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc1_2345,
        0xffc5_4321,
    ];

    let rounding_scalar = f32::from_bits(scalar_bits[0]);
    assert_eq!((upstream[0] / rounding_scalar).to_bits(), 0x4e13_d35b);
    assert_eq!(
        (upstream[0] * rounding_scalar.recip()).to_bits(),
        0x4e13_d35a
    );

    for scalar_bits in scalar_bits {
        let scalar = f32::from_bits(scalar_bits);
        let leaf = Tensor::ones([upstream.len()])
            .unwrap()
            .with_requires_grad(true);
        let weights = Tensor::from_vec(upstream.to_vec(), [upstream.len()]).unwrap();
        leaf.div_scalar(scalar)
            .unwrap()
            .mul(&weights)
            .unwrap()
            .sum()
            .backward()
            .unwrap();

        let actual = values(&leaf.grad().unwrap().unwrap())
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        let expected = upstream
            .iter()
            .map(|value| (value / scalar).to_bits())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "scalar bits {scalar_bits:#010x}");
    }

    let view_leaf = Tensor::ones([2, 3]).unwrap().with_requires_grad(true);
    let view = view_leaf.transpose(0, 1).unwrap();
    let view_upstream = [
        f32::from_bits(0x46d7_5128),
        -0.0,
        f32::INFINITY,
        -1.0,
        0.0,
        f32::NEG_INFINITY,
    ];
    let weights = Tensor::from_vec(view_upstream.to_vec(), [3, 2]).unwrap();
    let divided = view.div_scalar(rounding_scalar).unwrap();
    assert!(divided.requires_grad());
    assert_eq!(divided.stride(), [1, 3]);
    divided.mul(&weights).unwrap().sum().backward().unwrap();

    let actual = values(&view_leaf.grad().unwrap().unwrap())
        .into_iter()
        .map(f32::to_bits)
        .collect::<Vec<_>>();
    let expected =
        [0, 2, 4, 1, 3, 5].map(|index| (view_upstream[index] / rounding_scalar).to_bits());
    assert_eq!(actual, expected);

    let empty_leaf = Tensor::zeros([0]).unwrap().with_requires_grad(true);
    let empty = empty_leaf.reshape([2, 0, 3]).unwrap();
    let empty_output = empty.div_scalar(-0.0).unwrap();
    assert!(empty_output.requires_grad());
    assert_eq!(empty_output.shape(), [2, 0, 3]);
    assert_eq!(empty_output.stride(), [3, 3, 1]);
    empty_output.sum().backward().unwrap();
    let empty_gradient = empty_leaf.grad().unwrap().unwrap();
    assert_eq!(empty_gradient.shape(), [0]);
    assert!(values(&empty_gradient).is_empty());
}

#[test]
fn tensor_by_scalar_division_obeys_boundaries_and_graph_lifetime() {
    let leaf = Tensor::from_vec(vec![2.0, 3.0], [2])
        .unwrap()
        .with_requires_grad(true);
    let tracked = leaf.div_scalar(2.0).unwrap();
    assert!(tracked.requires_grad());

    let reflected = leaf.scalar_div(2.0).unwrap();
    assert!(!reflected.requires_grad());
    assert!(
        !leaf
            .detach()
            .unwrap()
            .div_scalar(2.0)
            .unwrap()
            .requires_grad()
    );
    {
        let _guard = no_grad();
        let suppressed = leaf.transpose(0, 0).unwrap().div_scalar(2.0).unwrap();
        assert!(!suppressed.requires_grad());
        assert_eq!(
            suppressed.sum().backward(),
            Err(TensorError::DoesNotRequireGrad)
        );
    }

    let no_edge_leaf = Tensor::from_vec(vec![4.0, 6.0], [2])
        .unwrap()
        .with_requires_grad(true);
    let no_edge_view = {
        let _guard = no_grad();
        no_edge_leaf.transpose(0, 0).unwrap()
    };
    assert!(no_edge_view.requires_grad());
    let no_edge_loss = no_edge_view.div_scalar(2.0).unwrap().sum();
    assert!(no_edge_loss.requires_grad());
    no_edge_loss.backward().unwrap();
    assert!(no_edge_leaf.grad().unwrap().is_none());
    assert_eq!(
        no_edge_loss.backward(),
        Err(TensorError::BackwardGraphFreed)
    );

    let loss = tracked.sum();
    loss.backward().unwrap();
    assert_eq!(values(&leaf.grad().unwrap().unwrap()), [0.5, 0.5]);
    assert_eq!(loss.backward(), Err(TensorError::BackwardGraphFreed));
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
