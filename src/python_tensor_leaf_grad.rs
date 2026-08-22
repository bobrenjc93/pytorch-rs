//! Python leaf-gradient descriptors and mutation for native tensors.

use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError},
    prelude::*,
    types::{PyAny, PyBool, PyDict, PyTuple},
};

use crate::{
    python::{
        PyTensor, PyTensorBase, dispatch_tensorbase_getset_mode, dispatch_tensorbase_method_mode,
        python_type_name,
    },
    python_dtype::dtype_object,
    python_tensor_errors::tensor_error,
};

const REQUIRES_GRAD_FALSE_ERROR: &str =
    "requires_grad_(False) is not supported; only enabling leaf tensors is implemented";

fn bind_requires_grad(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<bool> {
    if args.len() > 1 {
        return Err(PyTypeError::new_err(format!(
            "requires_grad_() takes from 0 to 1 positional arguments but {} were given",
            args.len()
        )));
    }

    let positional = if args.is_empty() {
        None
    } else {
        Some(args.get_item(0)?)
    };
    let keyword = kwargs
        .map(|kwargs| kwargs.get_item("requires_grad"))
        .transpose()?
        .flatten();
    let argument = positional.as_ref().or(keyword.as_ref());
    let requires_grad = if let Some(argument) = argument {
        if !argument.is_exact_instance_of::<PyBool>() {
            let position = if positional.is_some() {
                " (position 1)"
            } else {
                ""
            };
            let actual = python_type_name(argument)?;
            return Err(PyTypeError::new_err(format!(
                "requires_grad_(): argument 'requires_grad'{position} must be bool, not {actual}"
            )));
        }
        argument.is_truthy()?
    } else {
        true
    };

    if let Some(kwargs) = kwargs {
        for (key, _) in kwargs {
            let key = key.extract::<String>()?;
            if key != "requires_grad" {
                return Err(PyTypeError::new_err(format!(
                    "requires_grad_() got an unexpected keyword argument '{key}'"
                )));
            }
            if positional.is_some() {
                return Err(PyTypeError::new_err(
                    "requires_grad_() got multiple values for argument 'requires_grad'",
                ));
            }
        }
    }

    Ok(requires_grad)
}

#[pymethods]
impl PyTensorBase {
    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nrequires_grad_(requires_grad=True) -> Tensor\n\nChange if autograd should record operations on this tensor: sets this tensor's\n:attr:`requires_grad` attribute in-place. Returns this tensor.\n\n:func:`requires_grad_`'s main use case is to tell autograd to begin recording\noperations on a Tensor ``tensor``. If ``tensor`` has ``requires_grad=False``\n(because it was obtained through a DataLoader, or required preprocessing or\ninitialization), ``tensor.requires_grad_()`` makes it so that autograd will\nbegin to record operations on ``tensor``.\n\nArgs:\n    requires_grad (bool): If autograd should record operations on this tensor.\n        Default: ``True``.\n\nExample::\n\n    >>> # Let's say we want to preprocess some saved weights and use\n    >>> # the result as new weights.\n    >>> saved_weights = [0.1, 0.2, 0.3, 0.25]\n    >>> loaded_weights = torch.tensor(saved_weights)\n    >>> weights = preprocess(loaded_weights)  # some function\n    >>> weights\n    tensor([-0.5503,  0.4926, -2.1158, -0.8303])\n\n    >>> # Now, start to record operations done to weights\n    >>> weights.requires_grad_()\n    >>> out = weights.pow(2).sum()\n    >>> out.backward()\n    >>> weights.grad\n    tensor([-1.1007,  0.9853, -4.2316, -1.6606])\n\n"]
    #[pyo3(signature = (*args, **kwargs), text_signature = None)]
    fn requires_grad_(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let requires_grad = bind_requires_grad(args, kwargs)?;
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_method_mode(
            slf.py(),
            tensor,
            "requires_grad_",
            "torch.Tensor.requires_grad_",
            args,
            kwargs,
        )? {
            return Ok(result);
        }

        if !requires_grad {
            return Err(PyRuntimeError::new_err(REQUIRES_GRAD_FALSE_ERROR));
        }

        // All native tensors which lack autograd metadata are leaves. Build
        // the complete accumulator before changing any of the receiver's
        // fields, then preserve the exact Python wrapper and storage alias.
        tensor.try_borrow_mut()?.inner_mut().enable_requires_grad();
        Ok(tensor.clone().unbind().into_any())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nThe allowed dtype of :attr:``grad`` for this tensor.\n\n:attr:``grad_dtype`` can be set to a specific dtype or ``None``. By default,\n``t.grad_dtype == t.dtype``. When not None, the autograd engine casts\nincoming gradients to this dtype. This attribute is only accessible and\nsettable for leaf tensors.\n\n.. warning::\n    Use with caution. Diverging the dtypes of a tensor and its gradient may\n    break downstream systems that assume they match.\n\nExample::\n\n    >>> x = torch.tensor([1.0, 2.0], requires_grad=True)\n    >>> x.grad_dtype\n    torch.float32\n\n    >>> x.grad_dtype = torch.float16\n    >>> x.grad_dtype\n    torch.float16\n\n    >>> # Allow any gradient dtype\n    >>> x.grad_dtype = None\n    >>> x.grad_dtype\n"]
    #[getter]
    fn grad_dtype(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_getset_mode(slf.py(), tensor, "grad_dtype")? {
            return Ok(result);
        }

        let dtype = {
            let tensor = tensor.try_borrow()?;
            if !tensor.inner().is_leaf() {
                return Err(PyRuntimeError::new_err(
                    "grad_dtype can only be accessed on leaf tensors.",
                ));
            }
            tensor.inner().dtype()
        };
        Ok(dtype_object(slf.py(), dtype)?
            .clone_ref(slf.py())
            .into_any())
    }
}

#[pymethods]
impl PyTensor {
    #[getter]
    fn requires_grad(&self) -> bool {
        self.inner().requires_grad()
    }

    #[getter]
    fn is_leaf(&self) -> bool {
        self.inner().is_leaf()
    }

    #[getter]
    fn grad(&self, py: Python<'_>) -> PyResult<Option<Py<Self>>> {
        if let Some(gradient) = self.grad_cache().get(py) {
            return Ok(Some(gradient.clone_ref(py)));
        }
        let Some(inner) = self
            .inner()
            .live_grad()
            .map_err(|error| tensor_error(&error))?
        else {
            return Ok(None);
        };
        let gradient = self
            .grad_cache()
            .get_or_try_init(py, || Py::new(py, Self::new(inner)))?;
        Ok(Some(gradient.clone_ref(py)))
    }
}
