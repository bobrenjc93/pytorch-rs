//! Column-tiled rank-two leading reduction; distinct from pointwise Programs.
use super::super::{TensorError, c_void, runtime};
use super::{current_context, driver, jit_module::Module};
use crate::{
    pointwise_ir::invalid,
    tensor::leading_sum::{Divisor, LeadingSum},
};
use std::sync::Arc;

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
        validate_device(device, context, descriptor.column_certificate)?;
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
fn validate_device(device: usize, context: usize, columns: u64) -> Result<(), TensorError> {
    let _guard = runtime()?.guard(device)?;
    if current_context()? != context {
        return Err(invalid("host plan context mismatch"));
    }
    let driver = driver()?;
    let mut ordinal = 0;
    let mut properties = [0; 5];
    // SAFETY: documented CUDA attributes and writable integers under the device guard.
    unsafe {
        driver.check((driver.device)(&raw mut ordinal), "cuCtxGetDevice")?;
        for (value, attribute) in properties.iter_mut().zip([75, 76, 10, 16, 39]) {
            driver.check(
                (driver.attribute)(value, attribute, ordinal),
                "cuDeviceGetAttribute(leading sum)",
            )?;
        }
    }
    validate_device_properties(columns, properties)
}
fn validate_device_properties(columns: u64, properties: [i32; 5]) -> Result<(), TensorError> {
    let [major, minor, warp, multiprocessors, threads] = properties;
    if (major, minor, warp) != (9, 0, 32) || multiprocessors <= 0 || threads <= 0 {
        return Err(invalid(
            "leading sum requires Hopper cc 9.0, warp 32 and positive device properties",
        ));
    }
    let multiprocessors = u64::try_from(multiprocessors).expect("positive device property");
    let threads = u64::try_from(threads).expect("positive device property");
    if u128::from(columns) >= 64 * u128::from(multiprocessors)
        || 256 * u128::from(columns) >= 32 * u128::from(multiprocessors) * u128::from(threads)
    {
        return Err(invalid(
            "leading sum exceeds the device outer-reduction envelope",
        ));
    }
    Ok(())
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
    let segments = descriptor.segments();
    let reduce = format!(
        r"
        unsigned long long length = (rows + {segments}ull - 1ull) / {segments}ull;
        float value = 0.0f;
        for (unsigned int segment = 0; segment < {segments}u; ++segment) {{
            float p[8] = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
            for (unsigned long long k = 0; k < length; k += 64ull) {{
                #pragma unroll
                for (unsigned int j = 0; j < 8; ++j) {{
                    unsigned long long position = k + threadIdx.y + 8ull * j;
                    if (position < length) {{
                        unsigned long long row = segment * length + position;
                        float x = row < rows && column < columns ? input[row * columns + column] : 0.0f;
                        p[j] = __fadd_rn(p[j], x);
                    }}
                }}
            }}
            // d16 then d8, independently for each complete 32-lane half.
            lo[threadIdx.y][threadIdx.x] = __fadd_rn(__fadd_rn(p[0], p[2]), __fadd_rn(p[1], p[3]));
            hi[threadIdx.y][threadIdx.x] = __fadd_rn(__fadd_rn(p[4], p[6]), __fadd_rn(p[5], p[7]));
            __syncthreads();
            for (unsigned int step = 4; step != 0; step >>= 1) {{
                if (threadIdx.y < step) {{
                    lo[threadIdx.y][threadIdx.x] = __fadd_rn(lo[threadIdx.y][threadIdx.x], lo[threadIdx.y + step][threadIdx.x]);
                    hi[threadIdx.y][threadIdx.x] = __fadd_rn(hi[threadIdx.y][threadIdx.x], hi[threadIdx.y + step][threadIdx.x]);
                }}
                __syncthreads();
            }}
            if (threadIdx.y == 0) {{
                float total = __fadd_rn(lo[0][threadIdx.x], hi[0][threadIdx.x]);
                value = segment == 0 ? total : __fadd_rn(value, total);
            }}
            // All reads finish before the next segment reuses either plane.
            __syncthreads();
        }}
        if (threadIdx.y == 0 && column < columns) {{
            output[column] = {epilogue};
        }}
        __syncthreads();
"
    );
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
    __shared__ float lo[8][32];
    __shared__ float hi[8][32];
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
