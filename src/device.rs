use std::fmt::{Display, Formatter};

/// Native execution devices implemented by tensor storage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Device {
    /// Host CPU memory and kernels.
    #[default]
    Cpu,
}

impl Device {
    /// Returns the device ordinal, or [`None`] when this device has no ordinal.
    #[must_use]
    pub const fn index(self) -> Option<usize> {
        match self {
            Self::Cpu => None,
        }
    }

    /// Reports whether this device executes on host CPU memory.
    #[must_use]
    pub const fn is_cpu(self) -> bool {
        match self {
            Self::Cpu => true,
        }
    }

    /// Reports whether this device executes on a CUDA accelerator.
    #[must_use]
    pub const fn is_cuda(self) -> bool {
        match self {
            Self::Cpu => false,
        }
    }

    /// Reports whether this device executes on an Intel XPU accelerator.
    #[must_use]
    pub const fn is_xpu(self) -> bool {
        match self {
            Self::Cpu => false,
        }
    }

    /// Reports whether this device executes on an Apple MPS accelerator.
    #[must_use]
    pub const fn is_mps(self) -> bool {
        match self {
            Self::Cpu => false,
        }
    }

    /// Reports whether this device represents metadata-only tensor storage.
    #[must_use]
    pub const fn is_meta(self) -> bool {
        match self {
            Self::Cpu => false,
        }
    }
}

impl Display for Device {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => formatter.write_str("cpu"),
        }
    }
}
