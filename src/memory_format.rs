use std::fmt::{Display, Formatter};

/// Storage layouts accepted by tensor-copy operations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MemoryFormat {
    /// Retain the source tensor's supported layout metadata.
    #[default]
    Preserve,
    /// Produce canonical contiguous row-major strides.
    Contiguous,
    /// Four-dimensional channels-last layout.
    ChannelsLast,
    /// Five-dimensional channels-last layout.
    ChannelsLast3d,
}

impl Display for MemoryFormat {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Preserve => formatter.write_str("preserve_format"),
            Self::Contiguous => formatter.write_str("contiguous_format"),
            Self::ChannelsLast => formatter.write_str("channels_last"),
            Self::ChannelsLast3d => formatter.write_str("channels_last_3d"),
        }
    }
}
