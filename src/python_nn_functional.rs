//! Native bridges for Python neural-network functional operators.

use std::ffi::CString;
use std::fmt::Write as _;

use pyo3::exceptions::{
    PyMemoryError, PyNotImplementedError, PyRuntimeError, PyTypeError, PyUserWarning,
};
use pyo3::types::{PyAny, PyModule, PyString};
use pyo3::{IntoPyObjectExt, prelude::*};

use crate::{
    DType, Device, Tensor, TensorError, is_grad_enabled,
    python::PyTensor,
    python_argument_schema::{ArgumentSchema, parse_float_like_argument},
    python_tensor_errors::tensor_error,
    python_torch_function_mode,
};

const LINEAR_EXACT_TENSORS_ERROR: &str =
    "linear() only supports exact native Tensor input and weight operands";
const LINEAR_EXACT_BIAS_ERROR: &str =
    "linear() only supports an exact native Tensor bias or bias=None";
const MSE_LOSS_EXACT_TENSORS_ERROR: &str =
    "mse_loss() only supports exact native Tensor input and target operands";

const DROPOUT_METADATA: [DropoutMetadata; 6] = [
    DropoutMetadata {
        public_function: "dropout",
        operation: "dropout",
        inplace_operation: "dropout_",
        supports_tensor_probability: true,
        supports_probability_one: true,
        supported_ranks: None,
    },
    DropoutMetadata {
        public_function: "alpha_dropout",
        operation: "alpha_dropout",
        inplace_operation: "alpha_dropout_",
        supports_tensor_probability: false,
        supports_probability_one: false,
        supported_ranks: None,
    },
    DropoutMetadata {
        public_function: "feature_alpha_dropout",
        operation: "feature_alpha_dropout",
        inplace_operation: "feature_alpha_dropout_",
        supports_tensor_probability: true,
        supports_probability_one: true,
        supported_ranks: None,
    },
    DropoutMetadata {
        public_function: "dropout1d",
        operation: "feature_dropout",
        inplace_operation: "feature_dropout_",
        supports_tensor_probability: true,
        supports_probability_one: false,
        supported_ranks: Some(&[2, 3]),
    },
    DropoutMetadata {
        public_function: "dropout2d",
        operation: "feature_dropout",
        inplace_operation: "feature_dropout_",
        supports_tensor_probability: true,
        supports_probability_one: false,
        supported_ranks: Some(&[2, 3, 4]),
    },
    DropoutMetadata {
        public_function: "dropout3d",
        operation: "feature_dropout",
        inplace_operation: "feature_dropout_",
        supports_tensor_probability: true,
        supports_probability_one: true,
        supported_ranks: Some(&[5]),
    },
];

#[derive(Clone, Copy)]
struct DropoutMetadata {
    public_function: &'static str,
    operation: &'static str,
    inplace_operation: &'static str,
    supports_tensor_probability: bool,
    supports_probability_one: bool,
    supported_ranks: Option<&'static [usize]>,
}

struct DropoutSchema {
    input: ArgumentSchema,
    probability: ArgumentSchema,
    training: ArgumentSchema,
}

impl DropoutSchema {
    const fn new(metadata: DropoutMetadata, inplace: bool) -> Self {
        let operation = if inplace {
            metadata.inplace_operation
        } else {
            metadata.operation
        };
        Self {
            input: ArgumentSchema::new(operation, "input", 1, "Tensor"),
            probability: ArgumentSchema::new(operation, "p", 2, "float"),
            training: ArgumentSchema::new(operation, "train", 3, "bool"),
        }
    }
}

fn dropout_metadata(kind: &str) -> PyResult<DropoutMetadata> {
    DROPOUT_METADATA
        .iter()
        .copied()
        .find(|metadata| metadata.public_function == kind)
        .ok_or_else(|| PyRuntimeError::new_err(format!("unknown dropout kind: {kind}")))
}

fn format_probability(py: Python<'_>, probability: f64) -> PyResult<String> {
    if probability.is_nan() {
        return Ok(if probability.is_sign_negative() {
            "-nan".to_owned()
        } else {
            "nan".to_owned()
        });
    }
    PyModule::import(py, "builtins")?
        .getattr("format")?
        .call1((probability, ".6g"))?
        .extract()
}

#[pyfunction]
fn _nn_functional_dropout(
    py: Python<'_>,
    kind: &str,
    input: &Bound<'_, PyAny>,
    probability: &Bound<'_, PyAny>,
    training: &Bound<'_, PyAny>,
    inplace: bool,
) -> PyResult<Py<PyAny>> {
    // This private bridge mirrors the native operator's schema checks for the
    // supported deterministic cases. It deliberately owns no random state or
    // mutation.
    let metadata = dropout_metadata(kind)?;
    let schema = DropoutSchema::new(metadata, inplace);
    let Ok(tensor) = input.cast::<PyTensor>() else {
        return Err(schema.input.type_error(input)?);
    };
    if !metadata.supports_tensor_probability && probability.cast::<PyTensor>().is_ok() {
        // Keep alpha dropout's supported probability surface to real scalars;
        // standard dropout already owns the scalar-Tensor compatibility path.
        return Err(schema.probability.type_error(probability)?);
    }
    let probability = parse_float_like_argument(schema.probability, probability)?;
    let training = schema.training.parse_exact_bool(training)?;

    if !(0.0..=1.0).contains(&probability) {
        let probability = format_probability(py, probability)?;
        return Err(PyRuntimeError::new_err(format!(
            "dropout probability has to be between 0 and 1, but got {probability}"
        )));
    }

    let input_rank = tensor.try_borrow()?.inner().shape().len();
    if let Some(supported_ranks) = metadata.supported_ranks
        && !supported_ranks.contains(&input_rank)
    {
        let supported_ranks = supported_ranks
            .iter()
            .map(|rank| format!("rank-{rank}"))
            .collect::<Vec<_>>()
            .join(" or ");
        return Err(PyNotImplementedError::new_err(format!(
            "torch_rs.nn.functional.{} only supports {supported_ranks} inputs",
            metadata.public_function,
        )));
    }

    let input_is_empty = tensor.try_borrow()?.inner().numel() == 0;
    let is_unbatched_dropout1d = metadata.public_function == "dropout1d" && input_rank == 2;
    if is_unbatched_dropout1d && inplace {
        return Err(PyNotImplementedError::new_err(
            "torch_rs.nn.functional.dropout1d does not support inplace=True for rank-2 inputs",
        ));
    }
    if !training || probability == 0.0 || input_is_empty {
        if is_unbatched_dropout1d {
            // PyTorch routes an unbatched input through unsqueeze(0) and
            // squeeze(0), so even an identity feature-dropout call returns a
            // distinct view with a SqueezeBackward1 edge.
            let output = tensor
                .try_borrow()?
                .inner()
                .unsqueeze_front()
                .and_then(|input| input.squeeze_dim(0))
                .map_err(|error| tensor_error(&error))?;
            return PyTensor::new(output).into_py_any(py);
        }
        return Ok(tensor.clone().unbind().into_any());
    }

    if metadata.supports_probability_one && !inplace && probability.to_bits() == 1.0_f64.to_bits() {
        let output = tensor
            .try_borrow()?
            .inner()
            .mul_scalar(0.0)
            .map_err(|error| tensor_error(&error))?;
        return PyTensor::new(output).into_py_any(py);
    }

    Err(PyNotImplementedError::new_err(format!(
        "torch_rs.nn.functional.{} does not support sampling",
        metadata.public_function
    )))
}

#[pyfunction]
fn _nn_functional_dropout_tensor_autograd_suffix(input: &PyTensor) -> String {
    if !input.inner().requires_grad() {
        return String::new();
    }
    input.inner().grad_fn_name().map_or_else(
        || ", requires_grad=True".to_owned(),
        |name| format!(", grad_fn=<{name}>"),
    )
}

fn exact_linear_tensor<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyTensor>> {
    if !value.is_exact_instance_of::<PyTensor>() {
        return Err(PyTypeError::new_err(LINEAR_EXACT_TENSORS_ERROR));
    }
    Ok(value
        .cast::<PyTensor>()
        .expect("an exact PyTensor instance must downcast")
        .clone())
}

fn exact_linear_bias<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyTensor>> {
    if !value.is_exact_instance_of::<PyTensor>() {
        return Err(PyTypeError::new_err(LINEAR_EXACT_BIAS_ERROR));
    }
    Ok(value
        .cast::<PyTensor>()
        .expect("an exact PyTensor instance must downcast")
        .clone())
}

fn exact_mse_loss_tensor<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyTensor>> {
    if !value.is_exact_instance_of::<PyTensor>() {
        return Err(PyTypeError::new_err(MSE_LOSS_EXACT_TENSORS_ERROR));
    }
    Ok(value
        .cast::<PyTensor>()
        .expect("an exact PyTensor instance must downcast")
        .clone())
}

const TENSOR_SIZE_PREFIX: &str = "torch.Size([";
const TENSOR_SIZE_SUFFIX: &str = "])";
const MSE_WARNING_PREFIX: &str = "Using a target size (";
const MSE_WARNING_INFIX: &str = ") that is different to the input size (";
const MSE_WARNING_SUFFIX: &str = "). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.";

fn decimal_length(mut value: usize) -> usize {
    let mut length = 1;
    while value >= 10 {
        value /= 10;
        length += 1;
    }
    length
}

fn tensor_size_display_length(shape: &[usize]) -> Option<usize> {
    shape.iter().enumerate().try_fold(
        TENSOR_SIZE_PREFIX.len() + TENSOR_SIZE_SUFFIX.len(),
        |length, (position, dimension)| {
            length
                .checked_add(if position == 0 { 0 } else { ", ".len() })?
                .checked_add(decimal_length(*dimension))
        },
    )
}

fn push_tensor_size(output: &mut String, shape: &[usize]) -> PyResult<()> {
    output.push_str(TENSOR_SIZE_PREFIX);
    for (position, dimension) in shape.iter().enumerate() {
        if position != 0 {
            output.push_str(", ");
        }
        write!(output, "{dimension}")
            .map_err(|_| PyRuntimeError::new_err("unable to format mse_loss broadcast warning"))?;
    }
    output.push_str(TENSOR_SIZE_SUFFIX);
    Ok(())
}

fn warn_mse_loss_broadcast(
    py: Python<'_>,
    input_shape: &[usize],
    target_shape: &[usize],
) -> PyResult<()> {
    let capacity = MSE_WARNING_PREFIX
        .len()
        .checked_add(
            tensor_size_display_length(target_shape)
                .ok_or_else(|| PyMemoryError::new_err("mse_loss broadcast warning is too large"))?,
        )
        .and_then(|length| length.checked_add(MSE_WARNING_INFIX.len()))
        .and_then(|length| {
            tensor_size_display_length(input_shape)
                .and_then(|input_length| length.checked_add(input_length))
        })
        .and_then(|length| length.checked_add(MSE_WARNING_SUFFIX.len()))
        .ok_or_else(|| PyMemoryError::new_err("mse_loss broadcast warning is too large"))?;
    let capacity_with_nul = capacity
        .checked_add(1)
        .ok_or_else(|| PyMemoryError::new_err("mse_loss broadcast warning is too large"))?;
    let mut message = String::new();
    message
        .try_reserve_exact(capacity_with_nul)
        .map_err(|_| PyMemoryError::new_err("unable to allocate mse_loss broadcast warning"))?;
    message.push_str(MSE_WARNING_PREFIX);
    push_tensor_size(&mut message, target_shape)?;
    message.push_str(MSE_WARNING_INFIX);
    push_tensor_size(&mut message, input_shape)?;
    message.push_str(MSE_WARNING_SUFFIX);
    debug_assert_eq!(message.len(), capacity);
    message.push('\0');
    let message = CString::from_vec_with_nul(message.into_bytes()).map_err(|_| {
        PyRuntimeError::new_err("mse_loss broadcast warning unexpectedly contained a NUL byte")
    })?;
    PyErr::warn(py, &py.get_type::<PyUserWarning>(), &message, 2)
}

fn linear_bias_size_error(rows: usize, out_features: usize, bias_features: usize) -> PyErr {
    PyRuntimeError::new_err(format!(
        "The expanded size of the tensor ({out_features}) must match the existing size ({bias_features}) at non-singleton dimension 1.  Target sizes: [{rows}, {out_features}].  Tensor sizes: [{bias_features}]"
    ))
}

fn validate_linear_bias(input_rank: usize, bias: Option<&PyTensor>) -> PyResult<()> {
    if bias.is_some() && !matches!(input_rank, 1..=3) {
        return Err(PyNotImplementedError::new_err(
            "torch_rs.nn.functional.linear only supports bias for rank-1, rank-2, or rank-3 input",
        ));
    }
    if bias.is_some_and(|bias| bias.inner().shape().len() != 1) {
        return Err(PyNotImplementedError::new_err(
            "torch_rs.nn.functional.linear only supports a rank-1 bias tensor",
        ));
    }
    Ok(())
}

fn linear_rank_one(
    input: &Tensor,
    transposed_weight: &Tensor,
    bias: Option<&PyTensor>,
) -> Result<Tensor, TensorError> {
    input
        .unsqueeze_front()
        .and_then(|input| {
            bias.map_or_else(
                || input.matmul(transposed_weight),
                |bias| input.matmul_with_row_bias(transposed_weight, bias.inner()),
            )
        })
        .and_then(|output| output.squeeze_dim(0))
}

fn linear_rank_two(
    input: &Tensor,
    transposed_weight: &Tensor,
    bias: Option<&PyTensor>,
) -> Result<Tensor, TensorError> {
    bias.map_or_else(
        || input.matmul(transposed_weight),
        |bias| input.matmul_with_row_bias(transposed_weight, bias.inner()),
    )
}

fn linear_rank_three(
    input: &Tensor,
    weight: &Tensor,
    transposed_weight: &Tensor,
    bias: Option<&PyTensor>,
) -> PyResult<Result<Tensor, TensorError>> {
    let input_shape = input.shape();
    let weight_shape = weight.shape();
    // PyTorch's rank-3 by rank-2 matmul folds the leading dimensions
    // when they are stride-compatible, the input is empty, or the
    // matrix operand requires gradients. Otherwise its batched path
    // reports this layout-dependent inner-dimension error.
    let folded_input_layout = input.numel() == 0
        || input.stride()[1].checked_mul(input_shape[1]) == Some(input.stride()[0]);
    let folds_to_matrix = weight.requires_grad() || folded_input_layout;
    if !folds_to_matrix && input_shape[2] != weight_shape[1] {
        return Err(PyRuntimeError::new_err(format!(
            "Expected size for first two dimensions of batch2 tensor to be: [{}, {}] but got: [{}, {}].",
            input_shape[0], input_shape[2], input_shape[0], weight_shape[1]
        )));
    }
    if !folded_input_layout
        && input_shape[2] == weight_shape[1]
        && let Some(bias) = bias
        && bias.inner().shape()[0] != weight_shape[0]
        && bias.inner().shape()[0] != 1
    {
        return Err(PyRuntimeError::new_err(format!(
            "The size of tensor a ({}) must match the size of tensor b ({}) at non-singleton dimension 2",
            weight_shape[0],
            bias.inner().shape()[0]
        )));
    }
    let output_shape = [
        i64::try_from(input_shape[0])
            .map_err(|_| tensor_error(&TensorError::StrideCalculationOverflow))?,
        i64::try_from(input_shape[1])
            .map_err(|_| tensor_error(&TensorError::StrideCalculationOverflow))?,
        i64::try_from(weight_shape[0])
            .map_err(|_| tensor_error(&TensorError::StrideCalculationOverflow))?,
    ];
    if !folded_input_layout && let Some(bias) = bias {
        return Ok(input
            .flatten(0, 1)
            .and_then(|input| input.matmul(transposed_weight))
            .and_then(|output| output.reshape(output_shape))
            .and_then(|output| output.add(bias.inner())));
    }
    Ok(input
        .flatten(0, 1)
        .and_then(|input| {
            bias.map_or_else(
                || input.matmul(transposed_weight),
                |bias| input.matmul_with_row_bias(transposed_weight, bias.inner()),
            )
        })
        .and_then(|output| output.reshape(output_shape)))
}

fn linear_bias_target_rows(input_rank: usize, input: &Tensor) -> PyResult<usize> {
    match input_rank {
        1 => Ok(1),
        2 => Ok(input.shape()[0]),
        3 => input.shape()[0]
            .checked_mul(input.shape()[1])
            .ok_or_else(|| tensor_error(&TensorError::StrideCalculationOverflow)),
        _ => unreachable!("linear input rank was validated above"),
    }
}

fn resolve_linear_output(
    output: Result<Tensor, TensorError>,
    input_rank: usize,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&PyTensor>,
) -> PyResult<Tensor> {
    match output {
        Ok(output) => Ok(output),
        Err(TensorError::ShapeMismatch { .. }) if bias.is_some() => {
            let bias_features = bias
                .expect("only biased linear can report a bias shape mismatch")
                .inner()
                .shape()[0];
            let target_rows = linear_bias_target_rows(input_rank, input)?;
            Err(linear_bias_size_error(
                target_rows,
                weight.shape()[0],
                bias_features,
            ))
        }
        Err(error) => Err(tensor_error(&error)),
    }
}

#[pyfunction]
fn _nn_functional_linear(
    py: Python<'_>,
    input: &Bound<'_, PyAny>,
    weight: &Bound<'_, PyAny>,
    bias: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    if !python_torch_function_mode::is_empty() {
        return Err(PyTypeError::new_err(
            "linear() does not support an active TorchFunctionMode",
        ));
    }

    let input = exact_linear_tensor(input)?;
    let weight = exact_linear_tensor(weight)?;
    let bias = (!bias.is_none())
        .then(|| exact_linear_bias(bias))
        .transpose()?;
    let input = input.try_borrow()?;
    let weight = weight.try_borrow()?;
    let bias = bias.as_ref().map(Bound::try_borrow).transpose()?;
    let input_rank = input.inner().shape().len();
    if !matches!(input_rank, 1..=3) || weight.inner().shape().len() != 2 {
        return Err(PyNotImplementedError::new_err(
            "torch_rs.nn.functional.linear only supports rank-1, rank-2, or rank-3 input and rank-2 weight tensors",
        ));
    }
    validate_linear_bias(input_rank, bias.as_deref())?;
    if is_grad_enabled()
        && (input.inner().requires_grad()
            || weight.inner().requires_grad()
            || bias
                .as_ref()
                .is_some_and(|bias| bias.inner().requires_grad()))
    {
        return Err(PyRuntimeError::new_err(
            "linear(): autograd recording is not supported",
        ));
    }

    let transposed_weight = weight
        .inner()
        .transpose(0, 1)
        .map_err(|error| tensor_error(&error))?;
    let output = match input_rank {
        1 => linear_rank_one(input.inner(), &transposed_weight, bias.as_deref()),
        2 => linear_rank_two(input.inner(), &transposed_weight, bias.as_deref()),
        3 => linear_rank_three(
            input.inner(),
            weight.inner(),
            &transposed_weight,
            bias.as_deref(),
        )?,
        _ => unreachable!("linear input rank was validated above"),
    };
    let output = resolve_linear_output(
        output,
        input_rank,
        input.inner(),
        weight.inner(),
        bias.as_deref(),
    )?;
    PyTensor::new(output).into_py_any(py)
}

#[pyfunction]
fn _nn_functional_mse_loss(
    py: Python<'_>,
    input: &Bound<'_, PyAny>,
    target: &Bound<'_, PyAny>,
    size_average: &Bound<'_, PyAny>,
    reduce: &Bound<'_, PyAny>,
    reduction: &Bound<'_, PyAny>,
    weight: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    if !python_torch_function_mode::is_empty() {
        return Err(PyTypeError::new_err(
            "mse_loss() does not support an active TorchFunctionMode",
        ));
    }
    if !size_average.is_none() || !reduce.is_none() {
        return Err(PyNotImplementedError::new_err(
            "torch_rs.nn.functional.mse_loss only supports size_average=None and reduce=None",
        ));
    }
    let supports_reduction = reduction
        .cast::<PyString>()
        .ok()
        .and_then(|reduction| reduction.to_str().ok())
        .is_some_and(|reduction| reduction == "none");
    if !supports_reduction {
        return Err(PyNotImplementedError::new_err(
            "torch_rs.nn.functional.mse_loss only supports reduction='none'",
        ));
    }
    if !weight.is_none() {
        return Err(PyNotImplementedError::new_err(
            "torch_rs.nn.functional.mse_loss only supports weight=None",
        ));
    }

    let input = exact_mse_loss_tensor(input)?;
    let target = exact_mse_loss_tensor(target)?;
    let input = input.try_borrow()?;
    let target = target.try_borrow()?;
    if input.inner().dtype() != DType::Float32
        || target.inner().dtype() != DType::Float32
        || input.inner().device() != Device::Cpu
        || target.inner().device() != Device::Cpu
    {
        return Err(PyNotImplementedError::new_err(
            "torch_rs.nn.functional.mse_loss only supports CPU float32 tensors",
        ));
    }
    let input_shape = input.inner().shape();
    let target_shape = target.inner().shape();
    let shapes_match = input_shape == target_shape;
    let broadcasts_one_scalar =
        !shapes_match && (input_shape.is_empty() != target_shape.is_empty());
    if !shapes_match && !broadcasts_one_scalar {
        return Err(PyNotImplementedError::new_err(
            "torch_rs.nn.functional.mse_loss does not support broadcasting",
        ));
    }
    if broadcasts_one_scalar {
        warn_mse_loss_broadcast(py, input_shape, target_shape)?;
    }
    if is_grad_enabled() && (input.inner().requires_grad() || target.inner().requires_grad()) {
        return Err(PyRuntimeError::new_err(
            "mse_loss(): autograd recording is not supported",
        ));
    }

    let output = input
        .inner()
        .squared_difference(target.inner())
        .map_err(|error| tensor_error(&error))?;
    PyTensor::new(output).into_py_any(py)
}

pub(crate) fn add_nn_functional_bridges(module: &Bound<'_, PyModule>) -> PyResult<()> {
    for function in [
        wrap_pyfunction!(_nn_functional_dropout, module)?,
        wrap_pyfunction!(_nn_functional_dropout_tensor_autograd_suffix, module)?,
        wrap_pyfunction!(_nn_functional_linear, module)?,
        wrap_pyfunction!(_nn_functional_mse_loss, module)?,
    ] {
        let name = function.getattr("__name__")?;
        module.add_function(function.clone())?;
        module.getattr("__all__")?.call_method1("remove", (name,))?;
    }
    Ok(())
}
