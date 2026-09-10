//! Public unflatten binding over the shared native view implementation.

use super::{
    ParsedCallArgument, ProbedTorchFunctionOverride, PyTensor, call_torch_function_handler,
    dimension_swap_argument_type_error, extract_dimension_swap_dimension,
    insert_ordered_torch_function_override, is_dimension_swap_integer, is_not_implemented,
    movedim_dimension_unpack_error, normalize_dimension, parse_tensor_argument, position_suffix,
    probe_torch_function_override, python_number_index, python_type_name,
    resolve_torch_function_override, torch_function_dispatch_error_for_overrides,
    torch_function_mode_stack, try_push_size, try_size_vector,
    validate_torch_function_mode_handler, variable_function,
};
use crate::python_tensor_shape::{unflatten_native, validate_unflatten_storage};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyMemoryError, PyNotImplementedError, PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyList, PyTuple};

// Generated schemas read the stored list/tuple contents, bypassing subclass
// __len__, __getitem__, and __iter__ hooks in both validation and conversion.
enum NativeSizes<'py> {
    List(Bound<'py, PyList>),
    Tuple(Bound<'py, PyTuple>),
}

impl<'py> NativeSizes<'py> {
    fn from_value(value: &Bound<'py, PyAny>) -> Option<Self> {
        if let Ok(sizes) = value.cast::<PyList>() {
            Some(Self::List(sizes.clone()))
        } else {
            value
                .cast::<PyTuple>()
                .ok()
                .map(|sizes| Self::Tuple(sizes.clone()))
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::List(sizes) => sizes.len(),
            Self::Tuple(sizes) => sizes.len(),
        }
    }

    fn get_item(&self, index: usize) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Self::List(sizes) => sizes.get_item(index),
            Self::Tuple(sizes) => sizes.get_item(index),
        }
    }
}

pub(crate) fn unflatten_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let (arguments, overrides) = bind_arguments(args, kwargs)?;
    if !torch_function_mode_stack::is_empty() || !overrides.is_empty() {
        let function = variable_function(py, "unflatten")?;
        let types = PyTuple::new(py, overrides.iter().map(|item| item.dispatch_type.clone()))?;
        // Keep the original call and disable the top mode for the entire
        // dispatch attempt, as for other generated public variable functions.
        let active_mode = torch_function_mode_stack::pop();
        if let Some(mode) = active_mode.get() {
            validate_torch_function_mode_handler(mode.bind(py))?;
            let handler = mode.bind(py).getattr("__torch_function__")?;
            let result =
                call_torch_function_handler(py, &handler, &function, &types, args, kwargs)?;
            if !is_not_implemented(py, &result) {
                return Ok(result);
            }
        }
        for probed in &overrides {
            let handler = resolve_torch_function_override(py, probed)?;
            let result =
                call_torch_function_handler(py, &handler, &function, &types, args, kwargs)?;
            if !is_not_implemented(py, &result) {
                return Ok(result);
            }
        }
        return Err(torch_function_dispatch_error_for_overrides(
            py,
            "torch.unflatten",
            active_mode.get(),
            &overrides,
        )?);
    }

    let [input, dim, sizes] = arguments;
    if !input.value.is_exact_instance_of::<PyTensor>() {
        return Err(PyNotImplementedError::new_err(
            "unflatten(): only exact native CPU float32 Tensor inputs are supported",
        ));
    }
    let sizes =
        NativeSizes::from_value(&sizes.value).expect("native sizes were validated before dispatch");
    let mut parsed = try_size_vector(sizes.len())?;
    for index in 0..sizes.len() {
        let size = sizes.get_item(index)?;
        let indexed = python_number_index(&size).map_err(|_| {
            movedim_dimension_unpack_error("unflatten", "sizes", index + 1, &size)
                .unwrap_or_else(|error| error)
        })?;
        let value = indexed.extract::<i64>().map_err(|_| {
            PyTypeError::new_err(format!(
                "unflatten(): argument 'sizes' failed to unpack the object at pos {} with error \"Overflow when unpacking long long\"",
                index + 1
            ))
        })?;
        try_push_size(&mut parsed, value)?;
    }
    // PyTorch fully unpacks sizes before converting dim: __index__ on an
    // accepted dimension can mutate sizes, and sizes conversion errors win.
    let dim = extract_dimension_swap_dimension(&dim.value)?;
    // The generated top-level binding checks dim before the method's empty-
    // sizes guard. Scalar dimensions use the public [-1, 0] range here.
    let tensor = input.value.cast::<PyTensor>()?.try_borrow()?;
    if parsed.is_empty() {
        normalize_dimension(dim, tensor.inner.shape().len().max(1))?;
        return Err(PyRuntimeError::new_err(
            "unflatten: sizes must be non-empty",
        ));
    }
    validate_unflatten_storage(&tensor)?;
    unflatten_native(&tensor, dim, &parsed)?.into_py_any(py)
}

fn bind_arguments<'py>(
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<(
    [ParsedCallArgument<'py>; 3],
    Vec<ProbedTorchFunctionOverride<'py>>,
)> {
    let names = ["input", "dim", "sizes"];
    if args.len() > names.len() {
        return Err(PyTypeError::new_err(format!(
            "unflatten() takes 3 positional arguments but {} were given",
            args.len()
        )));
    }
    let mut arguments = Vec::with_capacity(3);
    let mut overrides = Vec::new();
    // Validate each schema slot before reporting a later missing argument.
    // Only input has the legacy aliases accepted by PyTorch 2.13.
    for (index, name) in names.iter().enumerate() {
        let value = if index < args.len() {
            Some(args.get_item(index)?)
        } else if let Some(kwargs) = kwargs {
            let mut value = kwargs.get_item(*name)?;
            if index == 0 && value.is_none() {
                for alias in ["x", "a", "x1"] {
                    value = kwargs.get_item(alias)?;
                    if value.is_some() {
                        break;
                    }
                }
            }
            value
        } else {
            None
        };
        let Some(value) = value else {
            let missing = &names[index..];
            let quoted = missing
                .iter()
                .map(|name| format!("\"{name}\""))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(PyTypeError::new_err(format!(
                "unflatten() missing {} required positional {}: {quoted}",
                missing.len(),
                if missing.len() == 1 {
                    "arguments"
                } else {
                    "argument"
                }
            )));
        };
        let argument = ParsedCallArgument {
            value,
            position: (index < args.len()).then_some(index + 1),
        };
        validate_argument(index, &argument, &mut overrides)?;
        arguments.push(argument);
    }
    if let Some(kwargs) = kwargs {
        let consumed = 3 - args.len();
        if kwargs.len() > consumed {
            for key in kwargs.keys() {
                let key = key.extract::<String>()?;
                let Some(index) = names.iter().position(|name| *name == key) else {
                    return Err(PyTypeError::new_err(format!(
                        "unflatten() got an unexpected keyword argument '{key}'"
                    )));
                };
                if index < args.len() {
                    return Err(PyTypeError::new_err(format!(
                        "unflatten() got multiple values for argument '{key}'"
                    )));
                }
            }
        }
    }
    Ok((
        arguments
            .try_into()
            .unwrap_or_else(|_| unreachable!("three schema arguments were bound")),
        overrides,
    ))
}

fn validate_argument<'py>(
    index: usize,
    argument: &ParsedCallArgument<'py>,
    overrides: &mut Vec<ProbedTorchFunctionOverride<'py>>,
) -> PyResult<()> {
    let value = &argument.value;
    if index == 0 && value.is_exact_instance_of::<PyTensor>() {
        return Ok(());
    }
    // Native schema types take precedence over their own override handlers.
    // Non-schema objects and individual sizes elements can still dispatch.
    if index == 1 && is_dimension_swap_integer(value)? {
        return Ok(());
    }
    if index == 2
        && let Some(sizes) = NativeSizes::from_value(value)
    {
        return validate_sizes(argument, &sizes, overrides);
    }
    if let Some(probed) = probe_torch_function_override(value) {
        overrides.try_reserve(1).map_err(|_| {
            PyMemoryError::new_err("unable to allocate unflatten dispatch operands")
        })?;
        return insert_ordered_torch_function_override(overrides, &probed);
    }
    if index == 0 {
        parse_tensor_argument("unflatten", "input", argument)?;
    } else if index == 1 {
        return Err(dimension_swap_argument_type_error(
            "unflatten",
            "dim",
            argument.position,
            "int",
            &python_type_name(value)?,
        ));
    } else {
        return Err(sizes_type_error(argument)?);
    }
    Ok(())
}

fn validate_sizes<'py>(
    argument: &ParsedCallArgument<'py>,
    sizes: &NativeSizes<'py>,
    overrides: &mut Vec<ProbedTorchFunctionOverride<'py>>,
) -> PyResult<()> {
    // As with reshape, the schema checks the first ordinary element;
    // remaining conversion errors occur only after override dispatch.
    for index in 0..sizes.len() {
        let size = sizes.get_item(index)?;
        if let Some(probed) = probe_torch_function_override(&size) {
            overrides.try_reserve(1).map_err(|_| {
                PyMemoryError::new_err("unable to allocate unflatten dispatch operands")
            })?;
            insert_ordered_torch_function_override(overrides, &probed)?;
        } else if index == 0
            && (size.is_instance_of::<PyBool>() || python_number_index(&size).is_err())
        {
            if argument.position.is_none() {
                return Err(sizes_type_error(argument)?);
            }
            return Err(PyTypeError::new_err(format!(
                "unflatten(): argument 'sizes'{} must be tuple of ints, but found element of type {} at pos 0",
                position_suffix(argument.position),
                python_type_name(&size)?
            )));
        }
    }
    Ok(())
}

fn sizes_type_error(argument: &ParsedCallArgument<'_>) -> PyResult<PyErr> {
    Ok(PyTypeError::new_err(format!(
        "unflatten(): argument 'sizes'{} must be tuple of ints, not {}",
        position_suffix(argument.position),
        python_type_name(&argument.value)?
    )))
}
