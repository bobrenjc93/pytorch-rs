use std::fmt::{Display, Formatter};
use std::mem::size_of;

/// Floating-point limits associated with a native scalar type.
#[cfg(any(feature = "python-bindings", test))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FloatingPointInfo {
    bits: usize,
    eps: f64,
    max: f64,
    min: f64,
    resolution: f64,
    smallest_normal: f64,
}

#[cfg(any(feature = "python-bindings", test))]
impl FloatingPointInfo {
    #[must_use]
    pub(crate) const fn bits(self) -> usize {
        self.bits
    }

    #[must_use]
    pub(crate) const fn eps(self) -> f64 {
        self.eps
    }

    #[must_use]
    pub(crate) const fn max(self) -> f64 {
        self.max
    }

    #[must_use]
    pub(crate) const fn min(self) -> f64 {
        self.min
    }

    #[must_use]
    pub(crate) const fn resolution(self) -> f64 {
        self.resolution
    }

    #[must_use]
    pub(crate) const fn smallest_normal(self) -> f64 {
        self.smallest_normal
    }
}

/// Native scalar types implemented by tensor storage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DType {
    /// IEEE 754 single-precision floating point.
    #[default]
    Float32,
}

impl DType {
    /// Returns the stable public name of this scalar type.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Float32 => "float32",
        }
    }

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

    /// Returns the corresponding real-valued scalar type.
    #[must_use]
    pub const fn to_real(self) -> Self {
        match self {
            Self::Float32 => Self::Float32,
        }
    }

    /// Returns floating-point limits for this scalar type.
    #[must_use]
    #[cfg(any(feature = "python-bindings", test))]
    pub(crate) const fn floating_point_info(self) -> FloatingPointInfo {
        match self {
            Self::Float32 => FloatingPointInfo {
                bits: size_of::<f32>() * 8,
                eps: f32::EPSILON as f64,
                max: f32::MAX as f64,
                min: f32::MIN as f64,
                resolution: 1.0e-6,
                smallest_normal: f32::MIN_POSITIVE as f64,
            },
        }
    }
}

impl Display for DType {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use super::DType;

    #[test]
    fn float32_floating_point_info_uses_native_limits() {
        let info = DType::Float32.floating_point_info();

        assert_eq!(info.bits(), size_of::<f32>() * 8);
        assert_eq!(info.eps().to_bits(), f64::from(f32::EPSILON).to_bits());
        assert_eq!(info.max().to_bits(), f64::from(f32::MAX).to_bits());
        assert_eq!(info.min().to_bits(), f64::from(f32::MIN).to_bits());
        assert_eq!(info.resolution().to_bits(), 1.0e-6_f64.to_bits());
        assert_eq!(
            info.smallest_normal().to_bits(),
            f64::from(f32::MIN_POSITIVE).to_bits()
        );
        assert_eq!(DType::Float32.name(), "float32");
    }
}
