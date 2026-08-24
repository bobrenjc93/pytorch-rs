//! Typed identities for Python-visible autograd nodes.

#[derive(Clone, Copy)]
#[cfg_attr(not(feature = "python-bindings"), allow(dead_code))]
pub(crate) enum AutogradNode {
    Add,
    Alias,
    Ceil,
    Clone,
    Exp,
    Floor,
    Multiply,
    Negate,
    Permute,
    Power,
    Relu,
    ReflectedSubtract,
    Select,
    Sin,
    Slice,
    Sigmoid,
    Sqrt,
    Squeeze,
    SqueezeDimension,
    SqueezeDimensions,
    Subtract,
    Sum,
    Tanh,
    MatrixTranspose,
    Copy,
    Transpose,
    Trunc,
    Unbind,
    Unsqueeze,
    View,
}

#[cfg(feature = "python-bindings")]
impl AutogradNode {
    pub(crate) const fn python_name(self) -> &'static str {
        match self {
            Self::Add => "AddBackward0",
            Self::Alias => "AliasBackward0",
            Self::Ceil => "CeilBackward0",
            Self::Clone => "CloneBackward0",
            Self::Exp => "ExpBackward0",
            Self::Floor => "FloorBackward0",
            Self::Multiply => "MulBackward0",
            Self::Negate => "NegBackward0",
            Self::Permute => "PermuteBackward0",
            Self::Power => "PowBackward0",
            Self::Relu => "ReluBackward0",
            Self::ReflectedSubtract => "RsubBackward1",
            Self::Select => "SelectBackward0",
            Self::Sin => "SinBackward0",
            Self::Slice => "SliceBackward0",
            Self::Sigmoid => "SigmoidBackward0",
            Self::Sqrt => "SqrtBackward0",
            Self::Squeeze => "SqueezeBackward0",
            Self::SqueezeDimension => "SqueezeBackward1",
            Self::SqueezeDimensions => "SqueezeBackward2",
            Self::Subtract => "SubBackward0",
            Self::Sum => "SumBackward0",
            Self::Tanh => "TanhBackward0",
            Self::MatrixTranspose => "TBackward0",
            Self::Copy => "ToCopyBackward0",
            Self::Transpose => "TransposeBackward0",
            Self::Trunc => "TruncBackward0",
            Self::Unbind => "UnbindBackward0",
            Self::Unsqueeze => "UnsqueezeBackward0",
            Self::View => "ViewBackward0",
        }
    }
}
