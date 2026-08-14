use std::fmt::{Display, Formatter};
use std::mem::size_of;

/// Native scalar types implemented by tensor storage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DType {
    /// IEEE 754 single-precision floating point.
    #[default]
    Float32,
}

impl DType {
    /// Returns the number of bytes used to store one scalar value.
    #[must_use]
    pub const fn element_size(self) -> usize {
        match self {
            Self::Float32 => size_of::<f32>(),
        }
    }

    /// Reports whether values of this scalar type are floating point.
    #[must_use]
    pub const fn is_floating_point(self) -> bool {
        match self {
            Self::Float32 => true,
        }
    }

    /// Reports whether values of this scalar type are complex.
    #[must_use]
    pub const fn is_complex(self) -> bool {
        match self {
            Self::Float32 => false,
        }
    }

    /// Reports whether values of this scalar type use a quantized representation.
    #[must_use]
    pub const fn is_quantized(self) -> bool {
        match self {
            Self::Float32 => false,
        }
    }

    /// Reports whether values of this scalar type are signed.
    #[must_use]
    pub const fn is_signed(self) -> bool {
        match self {
            Self::Float32 => true,
        }
    }
}

impl Display for DType {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Float32 => formatter.write_str("float32"),
        }
    }
}
