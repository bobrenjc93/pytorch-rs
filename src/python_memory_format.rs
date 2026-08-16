//! Python bindings for native memory formats.

use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;

use crate::MemoryFormat;

static PRESERVE_FORMAT: PyOnceLock<Py<PyMemoryFormat>> = PyOnceLock::new();
static CONTIGUOUS_FORMAT: PyOnceLock<Py<PyMemoryFormat>> = PyOnceLock::new();
static CHANNELS_LAST: PyOnceLock<Py<PyMemoryFormat>> = PyOnceLock::new();
static CHANNELS_LAST_3D: PyOnceLock<Py<PyMemoryFormat>> = PyOnceLock::new();

/// Python memory-format descriptor backed by a native [`MemoryFormat`].
#[pyclass(
    name = "memory_format",
    module = "torch_rs",
    frozen,
    eq,
    hash,
    skip_from_py_object
)]
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct PyMemoryFormat {
    inner: MemoryFormat,
}

impl PyMemoryFormat {
    pub(crate) const fn inner(&self) -> MemoryFormat {
        self.inner
    }
}

#[pymethods]
impl PyMemoryFormat {
    fn __repr__(&self) -> String {
        format!("torch.{}", self.inner)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn __reduce__(&self) -> &'static str {
        match self.inner {
            MemoryFormat::Preserve => "torch.preserve_format",
            MemoryFormat::Contiguous => "torch.contiguous_format",
            MemoryFormat::ChannelsLast => "torch.channels_last",
            MemoryFormat::ChannelsLast3d => "torch.channels_last_3d",
        }
    }
}

/// Returns the canonical Python descriptor for a native memory format.
pub(crate) fn memory_format_object(
    py: Python<'_>,
    memory_format: MemoryFormat,
) -> PyResult<&'static Py<PyMemoryFormat>> {
    let object = match memory_format {
        MemoryFormat::Preserve => &PRESERVE_FORMAT,
        MemoryFormat::Contiguous => &CONTIGUOUS_FORMAT,
        MemoryFormat::ChannelsLast => &CHANNELS_LAST,
        MemoryFormat::ChannelsLast3d => &CHANNELS_LAST_3D,
    };
    object.get_or_try_init(py, || {
        Py::new(
            py,
            PyMemoryFormat {
                inner: memory_format,
            },
        )
    })
}
