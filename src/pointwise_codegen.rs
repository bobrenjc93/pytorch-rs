//! Mechanical emission of validated VM words. No numerical decisions live here.
use super::{
    ADD, Address, CONSTANT, CONSTANT_COS, CONSTANT_SIN, COPY, COS, FLIP, FMA, Graph, INPUT, MUL,
    NEG, Program, RELU, SCALAR, SIN, STORE, SUB, TensorError, Write, invalid,
};

#[cfg(test)]
#[path = "pointwise_codegen_tests.rs"]
mod tests;

const MAX_INSTRUCTIONS: usize = 256;
const MAX_REGISTERS: usize = 128;
const MAX_SOURCE_BYTES: usize = 65_536;

pub(super) fn source(
    program: &Program,
    graph: &Graph,
    addresses: &[Address],
) -> Result<Option<String>, TensorError> {
    if addresses.len() != graph.inputs {
        return Err(invalid("pointwise address arity mismatch"));
    }
    if program.instruction_count() > MAX_INSTRUCTIONS || program.register_count() > MAX_REGISTERS {
        return Ok(None);
    }
    let mut source = String::from(
        "// torch_rs exact Program direct v1; float32, no fast math\n\
         extern \"C\" __global__ void torch_rs_pointwise(\n\
         const float* x0, const float* x1",
    );
    for index in 0..graph.outputs.len() {
        write!(source, ", float* out{index}").unwrap();
    }
    source.push_str(", unsigned long long n, const unsigned int* plan, unsigned long long instruction_count, float* scratch, unsigned long long scratch_stride");
    for index in 0..graph.scalar_count() {
        write!(source, ", float s{index}").unwrap();
    }
    source.push_str(") {\nconst unsigned long long worker = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\nfor (unsigned long long i = worker; i < n; i += (unsigned long long)blockDim.x * gridDim.x) {\n");
    for index in 0..program.register_count() {
        writeln!(source, "float r{index};").unwrap();
    }
    let flip = |value: String| format!("__uint_as_float(__float_as_uint({value}) ^ 0x80000000u)");
    for &[opcode, destination, a, b, c, flags] in program.instructions() {
        let expression = match opcode {
            INPUT => format!("x{a}[{}]", addresses[a as usize].source()),
            SCALAR => {
                if flags == 0 {
                    format!("s{a}")
                } else {
                    flip(format!("s{a}"))
                }
            }
            CONSTANT => format!("__uint_as_float(0x{a:08x}u)"),
            COPY => format!("r{a}"),
            ADD => format!("__fadd_rn(r{a}, r{b})"),
            SUB => format!("__fsub_rn(r{a}, r{b})"),
            MUL => format!("__fmul_rn(r{a}, r{b})"),
            NEG => format!("__fsub_rn(0.0f, r{a})"),
            FLIP => flip(format!("r{a}")),
            RELU => format!("(r{a} < 0.0f ? 0.0f : r{a})"),
            SIN => format!(
                "sinf((__float_as_uint(r{a}) & 0x7f800000u) == 0 ? __uint_as_float(__float_as_uint(r{a}) & 0x80000000u) : r{a})"
            ),
            COS => format!("cosf(r{a})"),
            CONSTANT_SIN => format!("(float)sin((double)r{a})"),
            CONSTANT_COS => format!("(float)cos((double)r{a})"),
            FMA => {
                let left = if flags & 1 == 0 {
                    format!("r{a}")
                } else {
                    flip(format!("r{a}"))
                };
                let addend = if flags & 4 == 0 {
                    format!("r{c}")
                } else {
                    "0.0f".into()
                };
                let addend = if flags & 2 == 0 { addend } else { flip(addend) };
                format!("fmaf({left}, r{b}, {addend})")
            }
            STORE => {
                writeln!(source, "out{destination}[i] = r{a};").unwrap();
                continue;
            }
            _ => return Err(invalid("invalid direct instruction")),
        };
        writeln!(source, "r{destination} = {expression};").unwrap();
    }
    source.push_str("}\n}\n");
    // Include the complete ABI, declarations, address expressions and trailer.
    Ok((source.len() <= MAX_SOURCE_BYTES).then_some(source))
}
