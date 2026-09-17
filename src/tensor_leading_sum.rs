//! Immutable leading-axis reduction plans; no invocation storage is retained.
use super::{Arc, Tensor, requires_grad_flag};
use crate::{
    cuda::{jit_module::Module, leading_sum::Kernel},
    pointwise_ir::{indexing::Layout, invalid},
    tensor_error::TensorError,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScalarKind {
    Boolean,
    Integer,
    Float,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Divisor {
    None,
    Constant {
        kind: ScalarKind,
        bits: u64,
    },
    Runtime {
        slot: usize,
        negative: bool,
    },
    Dimension {
        axis: usize,
        certificate: Option<u64>,
    },
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LeadingSum {
    pub(crate) axis: i64,
    pub(crate) keepdim: bool,
    pub(crate) row_certificate: Option<u64>,
    pub(crate) divisor: Divisor,
    pub(crate) scalar_count: usize,
    pub(crate) column_certificate: u64,
    pub(crate) row_hint: u64,
}
impl LeadingSum {
    pub(crate) fn validate(&self) -> Result<(), TensorError> {
        if !matches!(self.axis, 0 | -2) {
            return Err(invalid("leading sum requires axis 0 or -2"));
        }
        if !(65..=256).contains(&self.row_hint)
            || self
                .row_certificate
                .is_some_and(|rows| rows != self.row_hint)
        {
            return Err(invalid(
                "leading sum requires a row hint in 65..=256 matching its certificate",
            ));
        }
        if !(132..=256).contains(&self.column_certificate)
            || !self.column_certificate.is_multiple_of(4)
        {
            return Err(invalid(
                "leading sum requires exact columns in 132..=256 divisible by four",
            ));
        }
        if self.scalar_count > 64 {
            return Err(invalid("leading sum scalar count exceeds 64"));
        }
        match self.divisor {
            Divisor::Constant {
                kind: ScalarKind::Boolean,
                bits,
            } if ![0f64.to_bits(), 1f64.to_bits()].contains(&bits) => {
                return Err(invalid("invalid leading sum boolean scalar"));
            }
            Divisor::Constant {
                kind: ScalarKind::Integer,
                bits,
            } => {
                let value = f64::from_bits(bits);
                if !value.is_finite()
                    || value.fract() != 0.
                    || !(-9_223_372_036_854_775_808.0..=18_446_744_073_709_551_616.0)
                        .contains(&value)
                    || bits == (-0.0f64).to_bits()
                {
                    return Err(invalid("invalid leading sum integer scalar"));
                }
            }
            Divisor::Runtime { slot, .. } if slot >= self.scalar_count => {
                return Err(invalid("leading sum scalar slot out of range"));
            }
            Divisor::Dimension { axis, .. } if axis > 1 => {
                return Err(invalid("leading sum dimension axis out of range"));
            }
            _ => {}
        }
        if let Divisor::Dimension {
            axis: 0,
            certificate,
        } = self.divisor
            && certificate != self.row_certificate
        {
            return Err(invalid("leading sum row/divisor certificates disagree"));
        }
        if let Divisor::Dimension {
            axis: 1,
            certificate,
        } = self.divisor
            && certificate != Some(self.column_certificate)
        {
            return Err(invalid("leading sum column/divisor certificates disagree"));
        }
        Ok(())
    }
    pub(crate) fn segments(&self) -> u64 {
        self.row_hint.div_ceil(128)
    }
    pub(crate) fn scalar_count(&self) -> usize {
        self.scalar_count
    }
    pub(crate) fn validate_shape(&self, shape: &[usize]) -> Result<(), TensorError> {
        self.validate()?;
        if shape.len() != 2 {
            return Err(invalid("leading sum requires rank two"));
        }
        if !(65..=256).contains(&shape[0]) || shape[1] as u64 != self.column_certificate {
            return Err(invalid(
                "leading sum shape outside the certified row/column domain",
            ));
        }
        let check = |axis: usize, certificate: Option<u64>| {
            if certificate.map_or(shape[axis] >= 2, |exact| exact == shape[axis] as u64) {
                Ok(())
            } else {
                Err(invalid("leading sum dimension certificate mismatch"))
            }
        };
        check(0, self.row_certificate)?;
        if let Divisor::Dimension { axis, certificate } = self.divisor {
            check(axis, certificate)?;
        }
        Ok(())
    }
    pub(crate) fn identity(&self, device: usize, context: usize) -> Vec<u8> {
        // v2 fixes seeded 64-lane segments, separate B32 halves, u64 indices,
        // 32x8 geometry, no-FTZ full divide and the shared precise NVRTC options.
        let mut out = b"torch_rs.leading_sum.executable.v2\0".to_vec();
        let mut word = |x: u64| out.extend_from_slice(&x.to_le_bytes());
        for x in [
            device as u64,
            context as u64,
            self.axis.cast_unsigned(),
            u64::from(self.keepdim),
            self.scalar_count as u64,
            self.column_certificate,
            self.segments(),
        ] {
            word(x);
        }
        word(u64::from(self.row_certificate.is_some()));
        word(self.row_certificate.unwrap_or(0));
        match self.divisor {
            Divisor::None => word(0),
            Divisor::Constant { kind, bits } => {
                word(1);
                word(match kind {
                    ScalarKind::Boolean => 0,
                    ScalarKind::Integer => 1,
                    ScalarKind::Float => 2,
                });
                word(bits);
            }
            Divisor::Runtime { slot, negative } => {
                word(2);
                word(slot as u64);
                word(u64::from(negative));
            }
            Divisor::Dimension { axis, certificate } => {
                word(3);
                word(axis as u64);
                word(u64::from(certificate.is_some()));
                word(certificate.unwrap_or(0));
            }
        }
        out
    }
}

pub(crate) struct HostLeadingSum {
    descriptor: LeadingSum,
    device: usize,
    context: usize,
    input_shapes: Vec<Vec<usize>>,
    output: Layout,
    pub(crate) identity: Vec<u8>,
}
pub(crate) struct PreparedLeadingSum {
    kernel: Arc<Kernel>,
    input_shapes: Vec<Vec<usize>>,
    output: Layout,
}
impl HostLeadingSum {
    pub(crate) fn new(inputs: &[&Tensor], descriptor: LeadingSum) -> Result<Self, TensorError> {
        let device = validate_inputs(inputs, &descriptor)?;
        let context = Module::checked_context(device)?;
        let columns = inputs[0].shape[1];
        let output = Layout::new(&if descriptor.keepdim {
            vec![1, columns]
        } else {
            vec![columns]
        })?;
        Ok(Self {
            identity: descriptor.identity(device, context),
            descriptor,
            device,
            context,
            input_shapes: vec![inputs[0].shape.clone()],
            output,
        })
    }
    pub(crate) fn compile(&self) -> Result<Arc<Kernel>, TensorError> {
        Kernel::compile(self.descriptor.clone(), self.device, self.context)
    }
    pub(crate) fn bind(&self, kernel: Arc<Kernel>) -> Result<PreparedLeadingSum, TensorError> {
        if kernel.identity != self.identity {
            return Err(invalid("host plan executable identity mismatch"));
        }
        if Module::checked_context(self.device)? != kernel.context {
            return Err(invalid("generated kernel context mismatch"));
        }
        Ok(PreparedLeadingSum {
            kernel,
            input_shapes: self.input_shapes.clone(),
            output: Layout {
                shape: self.output.shape.clone(),
                strides: self.output.strides.clone(),
                elements: self.output.elements,
            },
        })
    }
}
fn validate_inputs(inputs: &[&Tensor], descriptor: &LeadingSum) -> Result<usize, TensorError> {
    let device = Tensor::validate_pointwise_inputs(inputs)?;
    if inputs.len() != 1 {
        return Err(invalid(
            "leading sum requires one original Tensor occurrence",
        ));
    }
    descriptor.validate_shape(&inputs[0].shape)?;
    Ok(device)
}
impl PreparedLeadingSum {
    pub(crate) fn kernel(&self) -> &Arc<Kernel> {
        &self.kernel
    }
    pub(crate) fn input_shapes(&self) -> &[Vec<usize>] {
        &self.input_shapes
    }
    pub(crate) fn retained_bytes(&self) -> usize {
        size_of::<Self>()
            .saturating_add(self.input_shapes.capacity() * size_of::<Vec<usize>>())
            .saturating_add(
                self.input_shapes
                    .iter()
                    .map(|s| s.capacity() * size_of::<usize>())
                    .sum::<usize>(),
            )
            .saturating_add(self.output.shape.capacity() * size_of::<usize>())
            .saturating_add(self.output.strides.capacity() * size_of::<usize>())
    }
    pub(crate) fn run(
        &self,
        inputs: &[&Tensor],
        scalars: &[f32],
    ) -> Result<Vec<Tensor>, TensorError> {
        self.kernel.validate_scalars(scalars)?;
        let device = validate_inputs(inputs, &self.kernel.descriptor)?;
        if device != self.kernel.device {
            return Err(invalid("kernel device guard mismatch"));
        }
        if inputs[0].shape != self.input_shapes[0] {
            return Err(invalid("prepared leading sum input shape guard mismatch"));
        }
        let input = inputs[0];
        let storage = input.storage.cuda_leading_sum(
            input.offset,
            [input.shape[0], input.shape[1]],
            &self.kernel,
            scalars,
        )?;
        Ok(vec![Tensor {
            storage: Arc::new(storage),
            shape: self.output.shape.clone(),
            strides: self.output.strides.clone(),
            offset: 0,
            elements: self.output.elements,
            output_nr: 0,
            leaf_requires_grad: requires_grad_flag(false),
            view_requires_grad: None,
            autograd: None,
        }])
    }
}
