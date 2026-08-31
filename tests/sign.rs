use pytorch_rs::Tensor;

const SIGN_EDGE_BITS: [u32; 20] = [
    0x0000_0000,
    0x8000_0000,
    0x0000_0001,
    0x8000_0001,
    0x007f_ffff,
    0x807f_ffff,
    0x0080_0000,
    0x8080_0000,
    0x3eaa_aaab,
    0xbeaa_aaab,
    0x3f80_0000,
    0xbf80_0000,
    0x7f7f_ffff,
    0xff7f_ffff,
    0x7f80_0000,
    0xff80_0000,
    0x7f81_2345,
    0xff81_2345,
    0x7fc1_2345,
    0xffc5_4321,
];

const EXPECTED_SIGN_EDGE_BITS: [u32; 20] = [
    0x0000_0000,
    0x0000_0000,
    0x3f80_0000,
    0xbf80_0000,
    0x3f80_0000,
    0xbf80_0000,
    0x3f80_0000,
    0xbf80_0000,
    0x3f80_0000,
    0xbf80_0000,
    0x3f80_0000,
    0xbf80_0000,
    0x3f80_0000,
    0xbf80_0000,
    0x3f80_0000,
    0xbf80_0000,
    0x0000_0000,
    0x0000_0000,
    0x0000_0000,
    0x0000_0000,
];

fn tensor_bits(tensor: &Tensor) -> Vec<u32> {
    tensor.logical_values().map(f32::to_bits).collect()
}

fn tensor_from_bits(bits: &[u32], shape: impl Into<Vec<usize>>) -> Tensor {
    Tensor::from_vec(bits.iter().copied().map(f32::from_bits).collect(), shape).unwrap()
}

fn expected_sign_bit(value: f32) -> u32 {
    if value == 0.0 || value.is_nan() {
        0.0_f32.to_bits()
    } else if value.is_sign_negative() {
        (-1.0_f32).to_bits()
    } else {
        1.0_f32.to_bits()
    }
}

fn expected_sign_bits(tensor: &Tensor) -> Vec<u32> {
    tensor
        .try_to_vec()
        .unwrap()
        .into_iter()
        .map(expected_sign_bit)
        .collect()
}

fn assert_sign_result(source: &Tensor, actual: &Tensor, expected_stride: &[usize]) {
    assert_eq!(actual.shape(), source.shape());
    assert_eq!(actual.stride(), expected_stride);
    assert_eq!(actual.storage_offset(), 0);
    assert_eq!(actual.dtype(), source.dtype());
    assert_eq!(actual.device(), source.device());
    assert!(!actual.shares_storage_with(source));
}

#[test]
fn sign_returns_pytorch_real_sign_bits_for_layouts_and_edges() {
    let base = Tensor::from_vec(
        (0_u8..24).map(|value| f32::from(value) - 12.0).collect(),
        [2, 3, 4],
    )
    .unwrap();
    let strided = base.transpose(0, 2).unwrap();

    let scalar = Tensor::from_vec(vec![-0.0], []).unwrap();
    let empty = Tensor::zeros([2, 0, 3])
        .unwrap()
        .transpose(0, 2)
        .unwrap()
        .index_integer(1)
        .unwrap();
    let contiguous = Tensor::from_vec(vec![-2.0, -0.0, 0.0, 3.0], [4]).unwrap();
    let offset = strided.index_integer(1).unwrap();
    let cases = [
        (
            "scalar",
            scalar.clone(),
            vec![],
            expected_sign_bits(&scalar),
        ),
        (
            "empty",
            empty.clone(),
            vec![2, 1],
            expected_sign_bits(&empty),
        ),
        (
            "contiguous",
            contiguous.clone(),
            vec![1],
            expected_sign_bits(&contiguous),
        ),
        (
            "offset",
            offset.clone(),
            vec![1, 3],
            expected_sign_bits(&offset),
        ),
        (
            "noncontiguous",
            strided.clone(),
            vec![1, 4, 12],
            expected_sign_bits(&strided),
        ),
        (
            "edges",
            tensor_from_bits(&SIGN_EDGE_BITS, [SIGN_EDGE_BITS.len()]),
            vec![1],
            EXPECTED_SIGN_EDGE_BITS.to_vec(),
        ),
    ];

    for (case, source, expected_stride, expected) in cases {
        let actual = source.sign().unwrap();
        assert_sign_result(&source, &actual, &expected_stride);
        assert_eq!(tensor_bits(&actual), expected, "{case}");
    }
}

#[test]
fn sign_records_first_order_zero_gradient_for_views_and_edges() {
    let leaf = tensor_from_bits(
        &[
            0xbf80_0000,
            0x8000_0000,
            0x0000_0000,
            0x3f80_0000,
            0x7f80_0000,
            0xff80_0000,
            0x7fc1_2345,
            0xffc5_4321,
        ],
        [2, 4],
    )
    .with_requires_grad(true);
    let source = leaf.transpose(0, 1).unwrap();
    let weights = Tensor::from_vec(
        vec![1.0, f32::NAN, f32::INFINITY, -0.0, 2.0, 3.0, 4.0, 5.0],
        [4, 2],
    )
    .unwrap();

    let output = source.sign().unwrap();
    assert!(output.requires_grad());
    assert!(!output.is_leaf());

    output.mul(&weights).unwrap().sum().backward().unwrap();
    let gradient = leaf.grad().unwrap().unwrap();
    assert_eq!(gradient.shape(), [2, 4]);
    assert_eq!(tensor_bits(&gradient), vec![0x0000_0000; 8]);
}
