//! A validated scalar program executed by one graph-keyed CUDA kernel.
//!
//! Numerical regions are planning units, not native launches. Instructions use
//! six native-endian u32 words, with no padding: opcode, destination, a, b, c,
//! flags. Operands are register indices except for input/scalar/constant loads.
//! Store's destination is an output slot. FMA flags encode negated product,
//! negated addend and a literal positive-zero addend in bits 0, 1 and 2.
use super::{Graph, indexing::Address, invalid};
use crate::tensor_error::TensorError;
use std::fmt::Write;

// Each branch of normalization interns at most eight expressions per original
// node, including converted constants and exposed signs. Contraction replaces
// an operation with one FMA; it does not expand the instruction stream.
const EXPANSION: usize = 8;
const INPUT: u32 = 0;
const SCALAR: u32 = 1;
const CONSTANT: u32 = 2;
const COPY: u32 = 3;
const ADD: u32 = 4;
const SUB: u32 = 5;
const MUL: u32 = 6;
const NEG: u32 = 7;
const FLIP: u32 = 8;
const RELU: u32 = 9;
const SIN: u32 = 10;
const COS: u32 = 11;
const CONSTANT_SIN: u32 = 12;
const CONSTANT_COS: u32 = 13;
const FMA: u32 = 14;
const STORE: u32 = 15;

#[derive(Clone, Debug)]
pub(super) enum Operation {
    Input(usize),
    Import(usize),
    Scalar(usize, bool),
    Constant(u32),
    Copy(usize),
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    Neg(usize),
    Flip(usize),
    Relu(usize),
    Sin(usize, bool),
    Cos(usize, bool),
    Fma {
        a: usize,
        b: usize,
        c: Option<usize>,
        negative_product: bool,
        negative_addend: bool,
    },
}

impl Operation {
    // This is a disassembly of the same operations encoded below, not another
    // numerical lowering. Region-local v numbers deliberately precede physical
    // register allocation so contractions and rounding are readable in audits.
    fn expression(&self, addresses: &[Address]) -> String {
        match *self {
            Self::Input(input) => format!("x{input}[{}]", addresses[input].source()),
            Self::Import(original) => format!("export{original}"),
            Self::Scalar(slot, negative) => format!("{}s{slot}", if negative { "-" } else { "" }),
            Self::Constant(bits) => format!("__uint_as_float(0x{bits:08x}u)"),
            Self::Copy(a) => format!("v{a}"),
            Self::Add(a, b) => format!("__fadd_rn(v{a}, v{b})"),
            Self::Sub(a, b) => format!("__fsub_rn(v{a}, v{b})"),
            Self::Mul(a, b) => format!("__fmul_rn(v{a}, v{b})"),
            Self::Neg(a) => format!("__fsub_rn(0.0f, v{a})"),
            Self::Flip(a) => format!("(-v{a})"),
            Self::Relu(a) => format!("(v{a} < 0.0f ? 0.0f : v{a})"),
            Self::Sin(a, true) => format!("(float)sin((double)v{a})"),
            Self::Cos(a, true) => format!("(float)cos((double)v{a})"),
            Self::Sin(a, false) => format!(
                "sinf((__float_as_uint(v{a}) & 0x7f800000u) == 0 ? __uint_as_float(__float_as_uint(v{a}) & 0x80000000u) : v{a})"
            ),
            Self::Cos(a, false) => format!("cosf(v{a})"),
            Self::Fma {
                a,
                b,
                c,
                negative_product,
                negative_addend,
            } => format!(
                "fmaf({}v{a}, v{b}, {}{})",
                if negative_product { "-" } else { "" },
                if negative_addend { "-" } else { "" },
                c.map_or_else(|| "0.0f".into(), |id| format!("v{id}")),
            ),
        }
    }

    fn encode(
        &self,
        destination: usize,
        assigned: &[usize],
        exports: &[Option<usize>],
    ) -> Result<[u32; 6], TensorError> {
        let reg = |id: usize| word(assigned[id]);
        let mut instruction = [0, word(destination)?, 0, 0, 0, 0];
        match *self {
            Self::Input(input) => {
                instruction[0] = INPUT;
                instruction[2] = word(input)?;
            }
            Self::Import(original) => {
                instruction[0] = COPY;
                instruction[2] =
                    word(exports[original].ok_or_else(|| invalid("missing numerical import"))?)?;
            }
            Self::Scalar(slot, negative) => {
                instruction[0] = SCALAR;
                instruction[2] = word(slot)?;
                instruction[5] = u32::from(negative);
            }
            Self::Constant(bits) => {
                instruction[0] = CONSTANT;
                instruction[2] = bits;
            }
            Self::Copy(a)
            | Self::Neg(a)
            | Self::Flip(a)
            | Self::Relu(a)
            | Self::Sin(a, _)
            | Self::Cos(a, _) => {
                instruction[0] = match *self {
                    Self::Copy(_) => COPY,
                    Self::Neg(_) => NEG,
                    Self::Flip(_) => FLIP,
                    Self::Relu(_) => RELU,
                    Self::Sin(_, false) => SIN,
                    Self::Sin(_, true) => CONSTANT_SIN,
                    Self::Cos(_, false) => COS,
                    Self::Cos(_, true) => CONSTANT_COS,
                    _ => unreachable!(),
                };
                instruction[2] = reg(a)?;
            }
            Self::Add(a, b) | Self::Sub(a, b) | Self::Mul(a, b) => {
                instruction[0] = match *self {
                    Self::Add(_, _) => ADD,
                    Self::Sub(_, _) => SUB,
                    Self::Mul(_, _) => MUL,
                    _ => unreachable!(),
                };
                instruction[2] = reg(a)?;
                instruction[3] = reg(b)?;
            }
            Self::Fma {
                a,
                b,
                c,
                negative_product,
                negative_addend,
            } => {
                instruction[0] = FMA;
                instruction[2] = reg(a)?;
                instruction[3] = reg(b)?;
                instruction[4] = c.map(reg).transpose()?.unwrap_or(0);
                instruction[5] = u32::from(negative_product)
                    | (u32::from(negative_addend) << 1)
                    | (u32::from(c.is_none()) << 2);
            }
        }
        Ok(instruction)
    }

    fn operands(&self) -> [Option<usize>; 3] {
        match *self {
            Self::Input(_) | Self::Import(_) | Self::Scalar(_, _) | Self::Constant(_) => [None; 3],
            Self::Copy(a)
            | Self::Neg(a)
            | Self::Flip(a)
            | Self::Relu(a)
            | Self::Sin(a, _)
            | Self::Cos(a, _) => [Some(a), None, None],
            Self::Add(a, b) | Self::Sub(a, b) | Self::Mul(a, b) => [Some(a), Some(b), None],
            Self::Fma { a, b, c, .. } => [Some(a), Some(b), c],
        }
    }
}

#[derive(Debug)]
pub(crate) struct Program {
    instructions: Vec<[u32; 6]>,
    register_count: usize,
    listing: Option<String>,
}

#[cfg(test)]
thread_local! {
    static BUILDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn build_count() -> usize {
    BUILDS.get()
}

#[derive(Default)]
struct Registers {
    count: usize,
    free: Vec<usize>,
}

impl Registers {
    fn allocate(&mut self) -> usize {
        self.free.pop().unwrap_or_else(|| {
            let register = self.count;
            self.count += 1;
            register
        })
    }
}

fn word(value: usize) -> Result<u32, TensorError> {
    u32::try_from(value).map_err(|_| invalid("numerical program index exceeds u32"))
}

// Following declarative FMA operands drops newly dead products without a
// second numerical pass. None means dead; roots remain live through export.
fn last_uses(operations: &[Operation], roots: &[usize]) -> Result<Vec<Option<usize>>, TensorError> {
    let mut last = vec![None; operations.len()];
    for &root in roots {
        let Some(use_at) = last.get_mut(root) else {
            return Err(invalid("invalid lowered numerical root"));
        };
        *use_at = Some(operations.len());
    }
    for id in (0..operations.len()).rev() {
        if last[id].is_some() {
            for operand in operations[id].operands().into_iter().flatten() {
                if operand >= id {
                    return Err(invalid("numerical operand must precede its consumer"));
                }
                last[operand] = Some(last[operand].unwrap_or(0).max(id));
            }
        }
    }
    Ok(last)
}

impl Program {
    pub(crate) fn build(
        graph: &Graph,
        addresses: &[Address],
        numerical_hint: u64,
        output_order: &[usize],
        scalar_output: bool,
    ) -> Result<Self, TensorError> {
        #[cfg(test)]
        BUILDS.set(BUILDS.get() + 1);
        Self::build_internal(
            graph,
            addresses,
            numerical_hint,
            output_order,
            scalar_output,
            None,
        )
    }

    /// Reconstruct a diagnostic listing through the same validated lowering.
    /// Normal execution does not allocate or format this text.
    pub(crate) fn describe(
        graph: &Graph,
        addresses: &[Address],
        numerical_hint: u64,
        output_order: &[usize],
        scalar_output: bool,
    ) -> Result<String, TensorError> {
        let program = Self::build_internal(
            graph,
            addresses,
            numerical_hint,
            output_order,
            scalar_output,
            Some(String::new()),
        )?;
        Ok(program.listing.expect("diagnostic listing was requested"))
    }

    fn build_internal(
        graph: &Graph,
        addresses: &[Address],
        numerical_hint: u64,
        output_order: &[usize],
        scalar_output: bool,
        listing: Option<String>,
    ) -> Result<Self, TensorError> {
        graph.validate()?;
        if addresses.len() != graph.inputs || output_order.len() != graph.outputs.len() {
            return Err(invalid("invalid numerical plan input or output count"));
        }
        let mut seen = vec![false; graph.outputs.len()];
        for &slot in output_order {
            let Some(mark) = seen.get_mut(slot) else {
                return Err(invalid("invalid numerical output order slot"));
            };
            if std::mem::replace(mark, true) {
                return Err(invalid("numerical output order must be a permutation"));
            }
        }
        let plan = super::regions::plan(
            graph,
            addresses,
            numerical_hint,
            output_order,
            scalar_output,
        );
        let nodes = graph.nodes.len();
        if plan.regions.is_empty() || plan.regions.len() > nodes {
            return Err(invalid("invalid numerical region count"));
        }
        let mut program = Self {
            instructions: Vec::new(),
            register_count: 0,
            listing,
        };
        let mut registers = Registers::default();
        let mut exports = vec![None; nodes];
        for (region_index, region) in plan.regions.into_iter().enumerate() {
            if let Some(listing) = &mut program.listing {
                writeln!(listing, "// numerical region {region_index}").unwrap();
            }
            program.append_region(
                graph,
                addresses,
                &plan.canonical,
                &region,
                &mut registers,
                &mut exports,
            )?;
        }
        if graph.outputs.iter().any(|&id| exports[id].is_none()) {
            return Err(invalid("numerical plan omitted a computed output"));
        }
        program.register_count = registers.count;
        program.validate(graph)?;
        Ok(program)
    }

    fn append_region(
        &mut self,
        graph: &Graph,
        addresses: &[Address],
        canonical: &super::regions::Canonical,
        region: &super::regions::Region,
        registers: &mut Registers,
        exports: &mut [Option<usize>],
    ) -> Result<(), TensorError> {
        let nodes = graph.nodes.len();
        if region.roots.is_empty() {
            return Err(invalid("numerical region has no roots"));
        }
        let mut imported = vec![false; nodes];
        for &id in &region.imports {
            if id >= nodes || exports[id].is_none() || imported[id] {
                return Err(invalid("numerical import must name an earlier export"));
            }
            imported[id] = true;
        }
        for &id in &region.roots {
            if id >= nodes || exports[id].is_some() {
                return Err(invalid(
                    "numerical region must export distinct original roots",
                ));
            }
        }
        let lowered = super::lowering::region(graph, canonical, &region.roots, &region.imports);
        let count = lowered.operations.len();
        if count > EXPANSION * nodes || lowered.roots.len() != region.roots.len() {
            return Err(invalid(
                "numerical normalization exceeded its structural bound",
            ));
        }
        let last = last_uses(&lowered.operations, &lowered.roots)?;
        let mut assigned = vec![usize::MAX; count];
        for (id, operation) in lowered.operations.iter().enumerate() {
            if last[id].is_none() {
                continue;
            }
            if let Some(listing) = &mut self.listing {
                writeln!(
                    listing,
                    "const float v{id} = {};",
                    operation.expression(addresses)
                )
                .unwrap();
            }
            let destination = registers.allocate();
            assigned[id] = destination;
            self.instructions
                .push(operation.encode(destination, &assigned, exports)?);
            for operand in operation.operands().into_iter().flatten() {
                if last[operand] == Some(id) && assigned[operand] != usize::MAX {
                    registers.free.push(assigned[operand]);
                    assigned[operand] = usize::MAX;
                }
            }
        }
        for (&original, &lowered) in region.roots.iter().zip(&lowered.roots) {
            if exports[original].is_some() {
                return Err(invalid("duplicate numerical export"));
            }
            let destination = registers.allocate();
            self.instructions
                .push([COPY, word(destination)?, word(assigned[lowered])?, 0, 0, 0]);
            exports[original] = Some(destination);
            if let Some(listing) = &mut self.listing {
                writeln!(listing, "export{original} = v{lowered};").unwrap();
            }
            if let Ok(slot) = graph.outputs.binary_search(&original) {
                self.instructions
                    .push([STORE, word(slot)?, word(destination)?, 0, 0, 0]);
                if let Some(listing) = &mut self.listing {
                    writeln!(listing, "out{slot}[i] = v{lowered};").unwrap();
                }
            }
        }
        // Export registers persist; the next region reuses every temporary.
        registers
            .free
            .extend(assigned.into_iter().filter(|&id| id != usize::MAX));
        Ok(())
    }

    pub(crate) fn instructions(&self) -> &[[u32; 6]] {
        &self.instructions
    }

    pub(crate) fn instruction_count(&self) -> usize {
        self.instructions.len()
    }

    pub(crate) fn register_count(&self) -> usize {
        self.register_count
    }

    fn validate(&self, graph: &Graph) -> Result<(), TensorError> {
        let nodes = graph.nodes.len();
        // N<=4096 makes these bounds representable even on a 32-bit host.
        let instruction_bound = EXPANSION * nodes * nodes + 2 * nodes + graph.outputs.len();
        let register_bound = (EXPANSION + 1) * nodes + graph.outputs.len();
        if self.instructions.is_empty()
            || self.instructions.len() > instruction_bound
            || self.register_count == 0
            || self.register_count > register_bound
        {
            return Err(invalid("numerical program exceeded its structural bounds"));
        }
        let mut initialized = vec![false; self.register_count];
        let mut stored = vec![false; graph.outputs.len()];
        for &[opcode, destination, a, b, c, flags] in &self.instructions {
            let read = |id: u32| {
                if initialized.get(id as usize) == Some(&true) {
                    Ok(())
                } else {
                    Err(invalid("numerical program reads an undefined register"))
                }
            };
            match opcode {
                INPUT if (a as usize) < graph.inputs && flags == 0 => {}
                SCALAR if (a as usize) < graph.scalar_count() && flags <= 1 => {}
                CONSTANT if flags == 0 => {}
                COPY | NEG | FLIP | RELU | SIN | COS | CONSTANT_SIN | CONSTANT_COS
                    if flags == 0 =>
                {
                    read(a)?;
                }
                ADD | SUB | MUL if flags == 0 => {
                    read(a)?;
                    read(b)?;
                }
                FMA if flags < 8 => {
                    read(a)?;
                    read(b)?;
                    if flags & 4 == 0 {
                        read(c)?;
                    }
                }
                STORE if flags == 0 => {
                    read(a)?;
                    let Some(slot) = stored.get_mut(destination as usize) else {
                        return Err(invalid("numerical program stores an invalid output"));
                    };
                    if std::mem::replace(slot, true) {
                        return Err(invalid("numerical program stores an output twice"));
                    }
                    continue;
                }
                _ => return Err(invalid("invalid numerical program opcode or operand")),
            }
            let Some(slot) = initialized.get_mut(destination as usize) else {
                return Err(invalid("numerical program writes an invalid register"));
            };
            *slot = true;
        }
        if stored.iter().any(|&value| !value) {
            return Err(invalid("numerical program has unwritten outputs"));
        }
        Ok(())
    }
}

/// The kernel contains no topology-dependent numerical decisions. A graph only
/// specializes its address maps and input/scalar/output ABI. Registers are
/// coalesced across workers: `scratch[register * scratch_stride + worker]`.
pub(super) fn source(graph: &Graph, addresses: &[Address]) -> String {
    let mut source = String::from(
        "// torch_rs typed pointwise scalar program v3; float32, no fast math\n\
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
    source.push_str(
        ") {\n\
         const unsigned long long worker = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n\
         float* registers = scratch + worker;\n\
         #define R(index) registers[(unsigned long long)(index) * scratch_stride]\n\
         for (unsigned long long i = worker; i < n; i += (unsigned long long)blockDim.x * gridDim.x) {\n\
         for (unsigned long long pc = 0; pc < instruction_count; ++pc) {\n\
         const unsigned int* instruction = plan + pc * 6;\n\
         const unsigned int opcode = instruction[0], d = instruction[1], a = instruction[2], b = instruction[3], c = instruction[4], flags = instruction[5];\n\
         switch (opcode) {\n",
    );
    writeln!(source, "case {INPUT}: switch (a) {{").unwrap();
    for (input, address) in addresses.iter().enumerate() {
        writeln!(
            source,
            "case {input}: R(d) = x{input}[{}]; break;",
            address.source()
        )
        .unwrap();
    }
    source.push_str("default: return; } break;\n");
    writeln!(source, "case {SCALAR}: {{ float value; switch (a) {{").unwrap();
    for slot in 0..graph.scalar_count() {
        writeln!(source, "case {slot}: value = s{slot}; break;").unwrap();
    }
    source.push_str("default: return; } R(d) = flags ? __uint_as_float(__float_as_uint(value) ^ 0x80000000u) : value; break; }\n");
    for (opcode, expression) in [
        (CONSTANT, "__uint_as_float(a)"),
        (COPY, "R(a)"),
        (ADD, "__fadd_rn(R(a), R(b))"),
        (SUB, "__fsub_rn(R(a), R(b))"),
        (MUL, "__fmul_rn(R(a), R(b))"),
        (NEG, "__fsub_rn(0.0f, R(a))"),
        (FLIP, "__uint_as_float(__float_as_uint(R(a)) ^ 0x80000000u)"),
        (RELU, "(R(a) < 0.0f ? 0.0f : R(a))"),
        (
            SIN,
            "sinf((__float_as_uint(R(a)) & 0x7f800000u) == 0 ? __uint_as_float(__float_as_uint(R(a)) & 0x80000000u) : R(a))",
        ),
        (COS, "cosf(R(a))"),
        (CONSTANT_SIN, "(float)sin((double)R(a))"),
        (CONSTANT_COS, "(float)cos((double)R(a))"),
    ] {
        writeln!(source, "case {opcode}: R(d) = {expression}; break;").unwrap();
    }
    writeln!(source, "case {FMA}: {{").unwrap();
    source.push_str(
        "float left = R(a), addend = (flags & 4u) ? 0.0f : R(c);\n\
         if (flags & 1u) left = __uint_as_float(__float_as_uint(left) ^ 0x80000000u);\n\
         if (flags & 2u) addend = __uint_as_float(__float_as_uint(addend) ^ 0x80000000u);\n\
         R(d) = fmaf(left, R(b), addend); break; }\n",
    );
    writeln!(source, "case {STORE}: switch (d) {{").unwrap();
    for slot in 0..graph.outputs.len() {
        writeln!(source, "case {slot}: out{slot}[i] = R(a); break;").unwrap();
    }
    source.push_str("default: return; } break;\ndefault: return;\n}\n}\n}\n#undef R\n}\n");
    source
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pointwise_ir::Node;

    fn graph() -> Graph {
        Graph {
            inputs: 1,
            nodes: vec![Node::Input(0), Node::Neg(0)],
            outputs: vec![1],
        }
    }

    #[test]
    fn verifier_checks_dataflow_output_coverage_and_flags() {
        let graph = graph();
        let mut program = Program {
            instructions: vec![
                [INPUT, 0, 0, 0, 0, 0],
                [NEG, 1, 0, 0, 0, 0],
                [STORE, 0, 1, 0, 0, 0],
            ],
            register_count: 2,
            listing: None,
        };
        assert!(program.validate(&graph).is_ok());
        program.instructions[1][2] = 1;
        assert!(program.validate(&graph).is_err());
        program.instructions[1][2] = 0;
        program.instructions[1][5] = 1;
        assert!(program.validate(&graph).is_err());
        program.instructions[1][5] = 0;
        program.instructions[2][1] = 1;
        assert!(program.validate(&graph).is_err());
        program.instructions.pop();
        assert!(program.validate(&graph).is_err());
    }

    #[test]
    fn output_order_is_validated_before_planning() {
        let graph = graph();
        for order in [vec![], vec![1], vec![0, 0]] {
            let execution =
                Program::build(&graph, &[Address::Linear], 1, &order, false).unwrap_err();
            let diagnostic =
                Program::describe(&graph, &[Address::Linear], 1, &order, false).unwrap_err();
            assert_eq!(execution.to_string(), diagnostic.to_string());
        }
    }

    #[test]
    fn listing_is_opt_in_without_changing_execution() {
        let multiple = Graph {
            inputs: 2,
            nodes: vec![
                Node::Input(0),
                Node::Input(1),
                Node::Mul(0, 1),
                Node::Neg(2),
                Node::Sin(2),
            ],
            outputs: vec![3, 4],
        };
        let check = |graph: &Graph, shapes: &[&[usize]]| {
            let indexing = graph.indexing(shapes).unwrap();
            let forward: Vec<_> = (0..graph.outputs.len()).collect();
            let reverse: Vec<_> = (0..graph.outputs.len()).rev().collect();
            for hint in [indexing.output.elements as u64, u64::MAX] {
                for order in [&forward, &reverse] {
                    let scalar = indexing.output.shape.is_empty();
                    let execution =
                        Program::build(graph, &indexing.addresses, hint, order, scalar).unwrap();
                    let diagnostic = Program::build_internal(
                        graph,
                        &indexing.addresses,
                        hint,
                        order,
                        scalar,
                        Some(String::new()),
                    )
                    .unwrap();
                    assert!(execution.listing.is_none());
                    assert_eq!(execution.instructions(), diagnostic.instructions());
                    assert_eq!(execution.register_count(), diagnostic.register_count());
                    assert_eq!(
                        Program::describe(graph, &indexing.addresses, hint, order, scalar).unwrap(),
                        diagnostic.listing.unwrap()
                    );
                }
            }
        };
        for graph in [graph(), multiple.clone()] {
            for shape in [vec![], vec![1], vec![2], vec![13], vec![257], vec![3, 4]] {
                check(&graph, &vec![shape.as_slice(); graph.inputs]);
            }
        }
        let broadcast = Graph {
            inputs: 2,
            nodes: vec![Node::Input(0), Node::Input(1), Node::Mul(0, 1)],
            outputs: vec![2],
        };
        check(&broadcast, &[&[3, 1], &[1, 4]]);
    }

    #[test]
    fn diagnostic_listing_keeps_its_exact_text() {
        let graph = Graph {
            inputs: 1,
            nodes: vec![Node::Input(0), Node::Relu(0)],
            outputs: vec![1],
        };
        assert_eq!(
            Program::describe(&graph, &[Address::Linear], 13, &[0], false).unwrap(),
            "// numerical region 0\n\
             const float v0 = x0[i];\n\
             const float v1 = (v0 < 0.0f ? 0.0f : v0);\n\
             export1 = v1;\n\
             out0[i] = v1;\n"
        );
    }

    #[test]
    fn diagnostic_listing_keeps_graph_and_address_validation() {
        let mut invalid_graph = graph();
        invalid_graph.nodes[0] = Node::Input(1);
        for (graph, addresses) in [
            (invalid_graph, vec![Address::Linear]),
            (graph(), vec![]),
            (graph(), vec![Address::Linear, Address::Linear]),
        ] {
            let execution = Program::build(&graph, &addresses, 1, &[0], false).unwrap_err();
            let diagnostic = Program::describe(&graph, &addresses, 1, &[0], false).unwrap_err();
            assert_eq!(execution.to_string(), diagnostic.to_string());
        }
    }

    #[test]
    fn maximum_live_graph_and_outputs_use_bounded_storage_and_one_kernel() {
        use crate::{
            Device, Tensor,
            cuda::jit::{Kernel, LaunchLayout},
        };

        // Every original node reaches a returned value. Runtime sine cannot be
        // replaced by the constant/alias simplifications of a negation chain.
        let mut nodes = vec![Node::Input(0)];
        for id in 1..4096 {
            nodes.push(Node::Sin(id - 1));
        }
        let graph = Graph {
            inputs: 1,
            nodes,
            outputs: (4032..4096).collect(),
        };
        assert_eq!(graph.nodes.len(), 4096);
        assert_eq!(graph.outputs.len(), 64);
        graph.validate().unwrap();
        let forward: Vec<_> = (0..64).collect();
        let reverse: Vec<_> = (0..64).rev().collect();
        for (label, order) in [("forward", &forward), ("reverse", &reverse)] {
            let program = Program::build(&graph, &[Address::Linear], 2, order, false).unwrap();
            assert!(
                program
                    .instructions()
                    .iter()
                    .filter(|instruction| instruction[0] == SIN)
                    .count()
                    >= 4095
            );
            assert_eq!(
                program
                    .instructions()
                    .iter()
                    .filter(|instruction| instruction[0] == STORE)
                    .count(),
                64
            );
            assert!(program.instruction_count() <= 8 * 4096 * 4096 + 2 * 4096 + 64);
            assert!(program.register_count() <= 9 * 4096 + 64);
            let layout = LaunchLayout::new(2, program.register_count()).unwrap();
            assert!(layout.scratch_elements * std::mem::size_of::<f32>() <= 64 * 1024 * 1024);
            eprintln!(
                "maximum pointwise graph: order={label}, nodes=4096, outputs=64, instructions={}, registers={}, scratch_bytes={}, workers={}",
                program.instruction_count(),
                program.register_count(),
                layout.scratch_elements * std::mem::size_of::<f32>(),
                layout.workers
            );
        }
        let mut oversized = graph.clone();
        oversized.nodes.push(Node::Sin(4095));
        oversized.outputs = (4033..4097).collect();
        assert!(
            oversized
                .validate()
                .unwrap_err()
                .to_string()
                .contains("4096 nodes")
        );
        assert!(Program::build(&oversized, &[Address::Linear], 2, &forward, false).is_err());

        if crate::cuda::device_count() == 0 {
            eprintln!(
                "skipping maximum-graph CUDA execution: CUDA unavailable; portable resource checks completed"
            );
            return;
        }
        let input = Tensor::from_vec(vec![0.0; 2], [2])
            .unwrap()
            .try_copy_cpu_to_cuda(Device::Cuda(0))
            .unwrap();
        // Compile exactly once. Different numerical order descriptors execute
        // through this same module/function and canonical output allocation ABI.
        let kernel = Kernel::compile(&graph, 0).unwrap();
        assert_eq!(kernel.ptx.matches(".visible .entry").count(), 1);
        let first =
            Tensor::pointwise_jit(&[&input], &kernel, &[], Some(2), Some(&forward)).unwrap();
        let second =
            Tensor::pointwise_jit(&[&input], &kernel, &[], Some(2), Some(&reverse)).unwrap();
        assert_eq!(first.len(), 64);
        assert_eq!(second.len(), 64);
        // Read both retained generations after the second call. Exact positive
        // zeros test every output store; all 128 allocations must remain distinct.
        let outputs: Vec<_> = first.iter().chain(&second).collect();
        for (index, output) in outputs.iter().enumerate() {
            assert_eq!(output.shape(), &[2]);
            assert!(!output.shares_storage_with(&input));
            for earlier in &outputs[..index] {
                assert!(!output.shares_storage_with(earlier));
            }
            let values = output.try_copy_cuda_to_cpu().unwrap().try_to_vec().unwrap();
            assert_eq!(values.len(), 2);
            assert!(
                values
                    .iter()
                    .all(|value| value.to_bits() == 0.0_f32.to_bits())
            );
        }
    }
}
