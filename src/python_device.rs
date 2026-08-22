//! Python bindings for native execution devices.

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyDict, PyInt, PyModule, PyString, PyTuple};

use crate::Device;

const UNINDEXED_DEVICE: i8 = -1;
const EXPECTED_DEVICE_TYPES: &str = "cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, maia, xla, lazy, vulkan, mps, meta, hpu, mtia, privateuseone";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RecognizedDeviceType {
    Cpu,
    Cuda,
    Ipu,
    Xpu,
    Mkldnn,
    OpenGl,
    OpenCl,
    Ideep,
    Hip,
    Ve,
    Fpga,
    Maia,
    Xla,
    Lazy,
    Vulkan,
    Mps,
    Meta,
    Hpu,
    Mtia,
    PrivateUseOne,
}

impl RecognizedDeviceType {
    pub(crate) const fn supports_autocast(self) -> bool {
        matches!(
            self,
            Self::Cpu
                | Self::Cuda
                | Self::Ipu
                | Self::Xpu
                | Self::Maia
                | Self::Xla
                | Self::Mps
                | Self::Hpu
                | Self::Mtia
                | Self::PrivateUseOne
        )
    }
}

fn normalize_device_index(index: i64) -> i8 {
    // PyTorch narrows Python's non-negative int64 index to its signed 8-bit
    // DeviceIndex, including the observable modulo-256 behavior.
    index.to_le_bytes()[0].cast_signed()
}

fn repr_device_index(index: i8) -> u16 {
    // PyTorch 2.13's repr widens a signed DeviceIndex directly to uint16_t.
    u16::from_le_bytes(i16::from(index).to_le_bytes())
}

fn device_string_error(mut message: String) -> PyErr {
    // PyTorch builds these diagnostics through a NUL-terminated C string, so
    // an embedded NUL truncates the observable message.
    if let Some(nul) = message.find('\0') {
        message.truncate(nul);
    }
    PyRuntimeError::new_err(message)
}

pub(crate) fn parse_device_specification(
    specification: &str,
) -> PyResult<(RecognizedDeviceType, Option<i32>)> {
    if specification.is_empty() {
        return Err(PyRuntimeError::new_err("Device string must not be empty"));
    }

    let (device_type, index) = specification
        .split_once(':')
        .map_or((specification, None), |(device_type, index)| {
            (device_type, Some(index))
        });
    let device_type = match device_type {
        "cpu" => RecognizedDeviceType::Cpu,
        "cuda" => RecognizedDeviceType::Cuda,
        "ipu" => RecognizedDeviceType::Ipu,
        "xpu" => RecognizedDeviceType::Xpu,
        "mkldnn" => RecognizedDeviceType::Mkldnn,
        "opengl" => RecognizedDeviceType::OpenGl,
        "opencl" => RecognizedDeviceType::OpenCl,
        "ideep" => RecognizedDeviceType::Ideep,
        "hip" => RecognizedDeviceType::Hip,
        "ve" => RecognizedDeviceType::Ve,
        "fpga" => RecognizedDeviceType::Fpga,
        "maia" => RecognizedDeviceType::Maia,
        "xla" => RecognizedDeviceType::Xla,
        "lazy" => RecognizedDeviceType::Lazy,
        "vulkan" => RecognizedDeviceType::Vulkan,
        "mps" => RecognizedDeviceType::Mps,
        "meta" => RecognizedDeviceType::Meta,
        "hpu" => RecognizedDeviceType::Hpu,
        "mtia" => RecognizedDeviceType::Mtia,
        "privateuseone" => RecognizedDeviceType::PrivateUseOne,
        _ if !device_type.is_empty()
            && device_type
                .bytes()
                .all(|byte| byte.is_ascii_alphabetic() || byte == b'_') =>
        {
            return Err(PyRuntimeError::new_err(format!(
                "Expected one of {EXPECTED_DEVICE_TYPES} device type at start of device string: {device_type}"
            )));
        }
        _ => {
            return Err(device_string_error(format!(
                "Invalid device string: '{specification}'"
            )));
        }
    };

    let index = index
        .map(|index| {
            let valid_digits = !index.is_empty() && index.bytes().all(|byte| byte.is_ascii_digit());
            if !valid_digits || (index.len() > 1 && index.starts_with('0')) {
                return Err(device_string_error(format!(
                    "Invalid device string: '{specification}'"
                )));
            }
            index.parse::<i32>().map_err(|_| {
                device_string_error(format!(
                    "Could not parse device index '{index}' in device string '{specification}'"
                ))
            })
        })
        .transpose()?;
    Ok((device_type, index))
}

/// Python device descriptor backed by a native [`Device`].
///
/// Tensor storage only implements ordinary CPU execution. The separate index
/// field preserves the descriptor metadata accepted by `torch.device` without
/// implying that indexed CPU execution exists in the native backend.
#[pyclass(name = "device", module = "torch_rs", frozen, eq, skip_from_py_object)]
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct PyDevice {
    inner: Device,
    index: i8,
}

impl PyDevice {
    pub(crate) const fn from_device(inner: Device) -> Self {
        Self {
            inner,
            index: UNINDEXED_DEVICE,
        }
    }

    const fn from_index(inner: Device, index: i8) -> Self {
        Self { inner, index }
    }

    pub(crate) const fn inner(&self) -> Device {
        self.inner
    }

    const fn has_index(&self) -> bool {
        self.index != UNINDEXED_DEVICE
    }
}

enum DeviceConstructorCall<'py> {
    Device(Bound<'py, PyAny>),
    TypeAndIndex {
        device_type: Bound<'py, PyAny>,
        index: Option<Bound<'py, PyAny>>,
        type_position: Option<usize>,
        index_position: Option<usize>,
    },
}

pub(crate) fn parse_device_value(function: &str, device: &Bound<'_, PyAny>) -> PyResult<Device> {
    parse_device_descriptor(function, device).map(|descriptor| descriptor.inner())
}

fn parse_device_descriptor(function: &str, device: &Bound<'_, PyAny>) -> PyResult<PyDevice> {
    if let Ok(device) = device.cast::<PyDevice>() {
        return Ok(device.try_borrow()?.clone());
    }
    if let Ok(device) = device.cast::<PyString>() {
        return parse_device_string(function, device.to_str()?);
    }

    let error = device_argument_type_error(function, device)?;
    Err(error)
}

fn parse_device_string(function: &str, specification: &str) -> PyResult<PyDevice> {
    if specification.is_empty() {
        return Err(PyRuntimeError::new_err("Device string must not be empty"));
    }

    let bytes = specification.as_bytes();
    let colon = bytes.iter().position(|byte| *byte == b':');
    let (device_type, index) = colon.map_or((specification, None), |position| {
        (
            &specification[..position],
            Some(&specification[position + 1..]),
        )
    });
    let valid_type = !device_type.is_empty()
        && device_type
            .bytes()
            .all(|byte| byte.is_ascii_alphabetic() || byte == b'_');
    let valid_index = index.is_none_or(|index| {
        !index.is_empty()
            && index.bytes().all(|byte| byte.is_ascii_digit())
            && (index.len() == 1 || !index.starts_with('0'))
    });
    if !valid_type || !valid_index {
        return Err(PyRuntimeError::new_err(format!(
            "Invalid device string: '{specification}'"
        )));
    }

    let index = match index {
        None => UNINDEXED_DEVICE,
        Some(index) => index.parse::<i32>().map_or_else(
            |_| {
                Err(PyRuntimeError::new_err(format!(
                    "Could not parse device index '{index}' in device string '{specification}'"
                )))
            },
            |index| Ok(normalize_device_index(i64::from(index))),
        )?,
    };

    if device_type == "cpu" {
        return Ok(PyDevice::from_index(Device::Cpu, index));
    }
    Err(PyRuntimeError::new_err(format!(
        "{function}(): device '{specification}' is not supported; only 'cpu' is implemented"
    )))
}

pub(crate) fn device_argument_type_error(
    function: &str,
    device: &Bound<'_, PyAny>,
) -> PyResult<PyErr> {
    let argument = if function == "device" {
        "type"
    } else {
        "device"
    };
    let type_name = device.get_type().name()?;
    Ok(PyTypeError::new_err(format!(
        "{function}(): argument '{argument}' must be torch.device or str, not {type_name}"
    )))
}

fn bind_device_constructor<'py>(
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<DeviceConstructorCall<'py>> {
    let mut type_keyword = None;
    let mut index_keyword = None;
    let mut device_keyword = None;
    if let Some(keywords) = keywords {
        for (key, value) in keywords {
            let key = key.extract::<String>()?;
            match key.as_str() {
                "type" => type_keyword = Some(value),
                "index" => index_keyword = Some(value),
                "device" => device_keyword = Some(value),
                _ => {
                    return Err(PyTypeError::new_err(format!(
                        "device() got an unexpected keyword argument '{key}'"
                    )));
                }
            }
        }
    }

    match positional.len() {
        0 => {
            if let Some(device_type) = type_keyword {
                if device_keyword.is_some() {
                    return Err(PyTypeError::new_err(
                        "device() got an unexpected keyword argument 'device'",
                    ));
                }
                return Ok(DeviceConstructorCall::TypeAndIndex {
                    device_type,
                    index: index_keyword,
                    type_position: None,
                    index_position: None,
                });
            }
            if let Some(device) = device_keyword {
                if index_keyword.is_some() {
                    return Err(PyTypeError::new_err(
                        "device() missing 1 required positional arguments: \"type\"",
                    ));
                }
                return Ok(DeviceConstructorCall::Device(device));
            }
            Err(invalid_device_constructor_arguments())
        }
        1 => {
            if type_keyword.is_some() {
                return Err(PyTypeError::new_err(
                    "device() got multiple values for argument 'type'",
                ));
            }
            if device_keyword.is_some() {
                return Err(PyTypeError::new_err(
                    "device() got an unexpected keyword argument 'device'",
                ));
            }
            let device_type = positional.get_item(0)?;
            if let Some(index) = index_keyword {
                Ok(DeviceConstructorCall::TypeAndIndex {
                    device_type,
                    index: Some(index),
                    type_position: Some(1),
                    index_position: None,
                })
            } else {
                Ok(DeviceConstructorCall::Device(device_type))
            }
        }
        2 if type_keyword.is_none() && index_keyword.is_none() && device_keyword.is_none() => {
            Ok(DeviceConstructorCall::TypeAndIndex {
                device_type: positional.get_item(0)?,
                index: Some(positional.get_item(1)?),
                type_position: Some(1),
                index_position: Some(2),
            })
        }
        _ => Err(invalid_device_constructor_arguments()),
    }
}

fn invalid_device_constructor_arguments() -> PyErr {
    PyTypeError::new_err(
        "device() received an invalid combination of arguments; expected a device or a type and optional index",
    )
}

fn constructor_type_name(value: &Bound<'_, PyAny>) -> PyResult<String> {
    if value.cast::<PyDevice>().is_ok() {
        return Ok("torch.device".to_owned());
    }
    let value_type = value.get_type();
    let name = value_type.name()?.to_string();
    let module = value_type.getattr("__module__")?.extract::<String>()?;
    Ok(if module == "numpy" {
        format!("numpy.{name}")
    } else {
        name
    })
}

fn validate_device_index_type(
    index: Option<&Bound<'_, PyAny>>,
    position: Option<usize>,
) -> PyResult<()> {
    let Some(index) = index else {
        return Ok(());
    };
    if index.is_none() {
        return Ok(());
    }
    let integer = !index.is_instance_of::<PyBool>() && index.is_instance_of::<PyInt>();
    let numpy_integer = if integer {
        false
    } else if let Ok(numpy) = PyModule::import(index.py(), "numpy") {
        index.is_instance(&numpy.getattr("integer")?)?
    } else {
        false
    };
    if integer || numpy_integer {
        return Ok(());
    }

    let type_name = constructor_type_name(index)?;
    let position = position.map_or_else(String::new, |position| format!(" (position {position})"));
    Err(PyTypeError::new_err(format!(
        "device(): argument 'index'{position} must be int, not {type_name}"
    )))
}

fn parse_explicit_device_index(index: Option<&Bound<'_, PyAny>>) -> PyResult<i8> {
    let Some(index) = index else {
        return Ok(UNINDEXED_DEVICE);
    };
    if index.is_none() {
        return Ok(UNINDEXED_DEVICE);
    }
    let index = index
        .extract::<i64>()
        .map_err(|_| PyValueError::new_err("Overflow when unpacking long long"))?;
    if index < 0 {
        return Err(PyRuntimeError::new_err("Device index must not be negative"));
    }
    Ok(normalize_device_index(index))
}

fn parse_type_and_index(
    device_type: &Bound<'_, PyAny>,
    index: Option<&Bound<'_, PyAny>>,
    type_position: Option<usize>,
    index_position: Option<usize>,
) -> PyResult<PyDevice> {
    let Ok(device_type) = device_type.cast::<PyString>() else {
        let type_name = constructor_type_name(device_type)?;
        let position =
            type_position.map_or_else(String::new, |position| format!(" (position {position})"));
        return Err(PyTypeError::new_err(format!(
            "device(): argument 'type'{position} must be str, not {type_name}"
        )));
    };
    validate_device_index_type(index, index_position)?;

    let specification = device_type.to_str()?;
    let parsed = parse_device_string("device", specification)?;
    if parsed.has_index() {
        return Err(PyRuntimeError::new_err(format!(
            "type (string) must not include an index because index was passed explicitly: {specification}"
        )));
    }
    let index = parse_explicit_device_index(index)?;
    Ok(PyDevice::from_index(parsed.inner(), index))
}

#[pymethods]
impl PyDevice {
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = None)]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        match bind_device_constructor(args, kwargs)? {
            DeviceConstructorCall::Device(device) => parse_device_descriptor("device", &device),
            DeviceConstructorCall::TypeAndIndex {
                device_type,
                index,
                type_position,
                index_position,
            } => parse_type_and_index(&device_type, index.as_ref(), type_position, index_position),
        }
    }

    #[getter]
    fn r#type(&self) -> &'static str {
        match self.inner {
            Device::Cpu => "cpu",
        }
    }

    #[getter]
    fn index(&self) -> Option<i8> {
        self.has_index().then_some(self.index)
    }

    fn __repr__(&self) -> String {
        if self.has_index() {
            format!(
                "device(type='{}', index={})",
                self.r#type(),
                repr_device_index(self.index)
            )
        } else {
            format!("device(type='{}')", self.r#type())
        }
    }

    fn __str__(&self) -> String {
        if self.has_index() {
            format!("{}:{}", self.r#type(), self.index)
        } else {
            self.r#type().to_owned()
        }
    }

    fn __hash__(&self) -> isize {
        isize::from(self.index.cast_unsigned())
    }

    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        let descriptor = slf.try_borrow()?;
        let arguments = if descriptor.has_index() {
            let device_type = PyString::new(py, descriptor.r#type()).into_any();
            let index = PyInt::new(py, i64::from(descriptor.index)).into_any();
            PyTuple::new(py, [device_type, index])?
        } else {
            PyTuple::new(py, [descriptor.r#type()])?
        };
        Ok((slf.getattr("__class__")?, arguments))
    }
}
