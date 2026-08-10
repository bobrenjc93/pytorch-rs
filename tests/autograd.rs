use pytorch_rs::{MemoryFormat, Tensor, TensorError, no_grad};

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
    }
    assert!(x.mul(&x).unwrap().requires_grad());
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
    assert_eq!(transposed.backward(), Err(TensorError::DoesNotRequireGrad));

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
