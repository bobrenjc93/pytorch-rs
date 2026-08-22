use std::fmt::{Display, Formatter};
use std::mem::size_of;

/// Native floating-point limits associated with a scalar type.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FloatingPointInfo {
    dtype: DType,
    bits: usize,
    resolution: f64,
    eps: f32,
    max: f32,
    min: f32,
    smallest_normal: f32,
    representation: &'static str,
}

impl FloatingPointInfo {
    #[must_use]
    pub const fn dtype(self) -> DType {
        self.dtype
    }

    #[must_use]
    pub const fn bits(self) -> usize {
        self.bits
    }

    #[must_use]
    pub const fn resolution(self) -> f64 {
        self.resolution
    }

    #[must_use]
    pub fn eps(self) -> f64 {
        f64::from(self.eps)
    }

    #[must_use]
    pub fn max(self) -> f64 {
        f64::from(self.max)
    }

    #[must_use]
    pub fn min(self) -> f64 {
        f64::from(self.min)
    }

    #[must_use]
    pub fn smallest_normal(self) -> f64 {
        f64::from(self.smallest_normal)
    }

    #[must_use]
    pub const fn representation(self) -> &'static str {
        self.representation
    }
}

const FLOAT32_INFO: FloatingPointInfo = FloatingPointInfo {
    dtype: DType::Float32,
    bits: size_of::<f32>() * u8::BITS as usize,
    resolution: 1.0e-6,
    eps: f32::EPSILON,
    max: f32::MAX,
    min: f32::MIN,
    smallest_normal: f32::MIN_POSITIVE,
    representation: "finfo(resolution=1e-06, min=-3.40282e+38, max=3.40282e+38, eps=1.19209e-07, smallest_normal=1.17549e-38, tiny=1.17549e-38, dtype=float32)",
};

/// Native scalar types implemented by tensor storage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DType {
    /// IEEE 754 single-precision floating point.
    #[default]
    Float32,
}

impl DType {
    /// Returns the compact scalar-type abbreviation used by `PyTorch`.
    #[must_use]
    pub const fn abbr(self) -> &'static str {
        match self {
            Self::Float32 => "f32",
        }
    }

    /// Returns the number of bytes used to store one scalar value.
    #[must_use]
    pub const fn element_size(self) -> usize {
        match self {
            Self::Float32 => size_of::<f32>(),
        }
    }

    /// Reports whether values of this scalar type can be cast to `to` under
    /// `PyTorch`'s casting rules.
    #[must_use]
    pub const fn can_cast_to(self, to: Self) -> bool {
        match (self, to) {
            (Self::Float32, Self::Float32) => true,
        }
    }

    /// Returns the scalar type produced by promoting this type with `other`.
    #[must_use]
    pub const fn promote(self, other: Self) -> Self {
        match (self, other) {
            (Self::Float32, Self::Float32) => Self::Float32,
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

    /// Returns the floating-point limits for this scalar type.
    #[must_use]
    pub const fn finfo(self) -> FloatingPointInfo {
        match self {
            Self::Float32 => FLOAT32_INFO,
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

#[cfg(test)]
mod tests {
    use super::DType;

    const CURRENT_DTYPES: [DType; 1] = [DType::Float32];

    #[test]
    fn casting_relation_covers_every_current_dtype_pair() {
        const EXPECTED: [[bool; 1]; 1] = [[true]];

        for (from_index, from) in CURRENT_DTYPES.into_iter().enumerate() {
            for (to_index, to) in CURRENT_DTYPES.into_iter().enumerate() {
                assert_eq!(
                    from.can_cast_to(to),
                    EXPECTED[from_index][to_index],
                    "casting {from} to {to}"
                );
            }
        }
    }

    #[test]
    fn promotion_relation_covers_every_current_dtype_pair() {
        const EXPECTED: [[DType; 1]; 1] = [[DType::Float32]];

        for (left_index, left) in CURRENT_DTYPES.into_iter().enumerate() {
            for (right_index, right) in CURRENT_DTYPES.into_iter().enumerate() {
                assert_eq!(
                    left.promote(right),
                    EXPECTED[left_index][right_index],
                    "promoting {left} with {right}"
                );
            }
        }
    }

    #[test]
    fn float32_limits_are_native_f32_metadata() {
        let info = DType::Float32.finfo();

        assert_eq!(info.dtype(), DType::Float32);
        assert_eq!(info.bits(), 32);
        assert_eq!(info.resolution().to_bits(), 1.0e-6_f64.to_bits());
        assert_eq!(info.eps().to_bits(), f64::from(f32::EPSILON).to_bits());
        assert_eq!(info.max().to_bits(), f64::from(f32::MAX).to_bits());
        assert_eq!(info.min().to_bits(), f64::from(f32::MIN).to_bits());
        assert_eq!(
            info.smallest_normal().to_bits(),
            f64::from(f32::MIN_POSITIVE).to_bits()
        );
        assert_eq!(
            info.representation(),
            "finfo(resolution=1e-06, min=-3.40282e+38, max=3.40282e+38, eps=1.19209e-07, smallest_normal=1.17549e-38, tiny=1.17549e-38, dtype=float32)"
        );
    }
}
