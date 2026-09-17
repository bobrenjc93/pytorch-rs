//! Column-tiled rank-two leading reduction; distinct from pointwise Programs.
use super::super::{TensorError, c_void};
use super::{driver, jit_module::Module};
use crate::{
    pointwise_ir::invalid,
    tensor::leading_sum::{Divisor, LeadingSum, Reduction},
};
use std::{fmt::Write, sync::Arc};

pub(crate) struct Kernel {
    pub(crate) descriptor: LeadingSum,
    pub(crate) identity: Vec<u8>,
    pub(crate) module: Module,
}
impl std::ops::Deref for Kernel {
    type Target = Module;
    fn deref(&self) -> &Module {
        &self.module
    }
}
impl Kernel {
    pub(crate) fn compile(
        descriptor: LeadingSum,
        device: usize,
        context: usize,
    ) -> Result<Arc<Self>, TensorError> {
        descriptor.validate()?;
        let source = source(&descriptor);
        let module = Module::compile(
            source,
            device,
            Some(context),
            false,
            c"torch_rs_leading_sum",
        )?;
        Ok(Arc::new(Self {
            identity: descriptor.identity(device, context),
            descriptor,
            module,
        }))
    }
    pub(crate) fn validate_scalars(&self, scalars: &[f32]) -> Result<(), TensorError> {
        if scalars.len() != self.descriptor.scalar_count() {
            return Err(invalid("leading sum runtime scalar arity mismatch"));
        }
        Ok(())
    }
    /// Input and output ranges are checked by storage; all owners remain live
    /// until the caller completes the legacy stream, even after launch error.
    #[allow(clippy::cast_precision_loss)] // Exact-dimension policy explicitly converts the current u64 extent to binary64.
    pub(crate) unsafe fn launch(
        &self,
        mut input: u64,
        mut output: u64,
        shape: [usize; 2],
        scalars: &[f32],
    ) -> Result<(), TensorError> {
        self.validate_scalars(scalars)?;
        self.validate_context()?;
        self.descriptor.validate_shape(&shape)?;
        let mut rows = shape[0] as u64;
        let mut columns = shape[1] as u64;
        let mut scalar = match self.descriptor.divisor {
            Divisor::Runtime { slot, negative } => {
                if negative {
                    -scalars[slot]
                } else {
                    scalars[slot]
                }
            }
            Divisor::Dimension {
                axis,
                certificate: Some(_),
            } => reciprocal(shape[axis] as f64),
            _ => 0.,
        };
        let mut args = [
            (&raw mut input).cast(),
            (&raw mut output).cast(),
            (&raw mut rows).cast(),
            (&raw mut columns).cast(),
            (&raw mut scalar).cast(),
        ];
        let driver = driver()?;
        // SAFETY: checked scalar/pointer ABI and storage lifetime above.
        driver.check(
            unsafe {
                (driver.launch)(
                    self.function as *mut c_void,
                    blocks(shape[1]),
                    1,
                    1,
                    32,
                    8,
                    1,
                    0,
                    std::ptr::without_provenance_mut(1),
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )
    }
}
pub(crate) fn blocks(columns: usize) -> u32 {
    u32::try_from(columns.div_ceil(32).min(65535)).expect("capped grid")
}
#[allow(clippy::cast_possible_truncation)] // Versioned policy rounds the binary64 reciprocal to binary32, including overflow.
fn reciprocal(value: f64) -> f32 {
    (1.0 / value) as f32
}
fn source(descriptor: &LeadingSum) -> String {
    let epilogue = match descriptor.divisor {
        Divisor::None => "value".into(),
        Divisor::Constant { bits, .. } => {
            let coefficient = reciprocal(f64::from_bits(bits));
            format!(
                "__fmul_rn(value, __uint_as_float({}u))",
                coefficient.to_bits()
            )
        }
        Divisor::Runtime { .. } => "full_divide(value, scalar)".into(),
        Divisor::Dimension {
            certificate: Some(_),
            ..
        } => "__fmul_rn(value, scalar)".into(),
        Divisor::Dimension {
            axis,
            certificate: None,
        } => format!(
            "full_divide(value, __ull2float_rn({}))",
            if axis == 0 { "rows" } else { "columns" }
        ),
    };
    let body = match descriptor.reduction() {
        Reduction::Empty => "float value = 0.0f;".into(),
        Reduction::Singleton => "float value = input[column];".into(),
        Reduction::OrderedSmall(rows) => {
            let mut s = "float value = input[column];\n".to_owned();
            for row in 1..rows {
                writeln!(
                    s,
                    "value = __fadd_rn(value, input[{row}ull * columns + column]);"
                )
                .unwrap();
            }
            s
        }
        Reduction::Tree8 => String::new(),
    };
    let reduce = if descriptor.reduction() == Reduction::Tree8 {
        format!(
            r"
        float partial = 0.0f;
        if (column < columns) {{
            for (unsigned long long row = threadIdx.y; row < rows; row += 8ull)
                partial = __fadd_rn(partial, input[row * columns + column]);
        }}
        workspace[threadIdx.y][threadIdx.x] = partial;
        __syncthreads();
        for (unsigned int step = 4; step != 0; step >>= 1) {{
            if (threadIdx.y < step)
                workspace[threadIdx.y][threadIdx.x] = __fadd_rn(workspace[threadIdx.y][threadIdx.x], workspace[threadIdx.y + step][threadIdx.x]);
            __syncthreads();
        }}
        if (threadIdx.y == 0 && column < columns) {{
            float value = workspace[0][threadIdx.x];
            output[column] = {epilogue};
        }}
        __syncthreads();
"
        )
    } else {
        format!(
            "if (threadIdx.y == 0 && column < columns) {{\n{body}\noutput[column] = {epilogue};\n}}"
        )
    };
    let division_helper = if matches!(
        descriptor.divisor,
        Divisor::Runtime { .. }
            | Divisor::Dimension {
                certificate: None,
                ..
            }
    ) {
        r#"extern "C" __device__ __forceinline__ float full_divide(float a, float b) {
    float out;
    asm("div.full.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
    return out;
}"#
    } else {
        ""
    };
    format!(
        r#"
{division_helper}
extern "C" __global__ void torch_rs_leading_sum(const float* input, float* output, unsigned long long rows, unsigned long long columns, float scalar) {{
    __shared__ float workspace[8][32];
    for (unsigned long long tile = (unsigned long long)blockIdx.x * 32ull; tile < columns; tile += (unsigned long long)gridDim.x * 32ull) {{
        unsigned long long column = tile + threadIdx.x;
        {reduce}
    }}
}}
"#
    )
}

#[cfg(test)]
#[path = "leading_sum_tests.rs"]
pub(crate) mod tests;
