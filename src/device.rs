use std::fmt::{Display, Formatter};

/// Native execution devices implemented by tensor storage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Device {
    /// Host CPU memory and kernels.
    #[default]
    Cpu,
    /// NVIDIA CUDA device memory.
    Cuda(usize),
}

impl Device {
    /// Returns the device ordinal, or [`None`] when this device has no ordinal.
    #[must_use]
    pub const fn index(self) -> Option<usize> {
        match self {
            Self::Cpu => None,
            Self::Cuda(index) => Some(index),
        }
    }

    /// Reports whether this device executes on host CPU memory.
    #[must_use]
    pub const fn is_cpu(self) -> bool {
        matches!(self, Self::Cpu)
    }

    /// Reports whether this device executes on a CUDA accelerator.
    #[must_use]
    pub const fn is_cuda(self) -> bool {
        matches!(self, Self::Cuda(_))
    }

    /// Reports whether this device executes on a Graphcore IPU accelerator.
    #[must_use]
    pub const fn is_ipu(self) -> bool {
        false
    }

    /// Reports whether this device executes on a Meta MTIA accelerator.
    #[must_use]
    pub const fn is_mtia(self) -> bool {
        false
    }

    /// Reports whether this device executes on a Maia accelerator.
    #[must_use]
    pub const fn is_maia(self) -> bool {
        false
    }

    /// Reports whether this device executes on an Intel XPU accelerator.
    #[must_use]
    pub const fn is_xpu(self) -> bool {
        false
    }

    /// Reports whether this device executes on an XLA accelerator.
    #[must_use]
    pub const fn is_xla(self) -> bool {
        false
    }

    /// Reports whether this device executes on an Apple MPS accelerator.
    #[must_use]
    pub const fn is_mps(self) -> bool {
        false
    }

    /// Reports whether this device executes on a Vulkan accelerator.
    #[must_use]
    pub const fn is_vulkan(self) -> bool {
        false
    }

    /// Reports whether this device represents metadata-only tensor storage.
    #[must_use]
    pub const fn is_meta(self) -> bool {
        false
    }
}

impl Display for Device {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => formatter.write_str("cpu"),
            Self::Cuda(index) => write!(formatter, "cuda:{index}"),
        }
    }
}
