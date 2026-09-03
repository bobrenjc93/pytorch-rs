import subprocess
import sys
import unittest
from dataclasses import FrozenInstanceError, dataclass
from types import SimpleNamespace

import torch_rs as torch
from torch_rs import _compile_bytecode
from torch_rs import _compile_trace

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


REFERENCE_PYTORCH_VERSION = "2.13.0"
COMPILE_CORPUS_VERSION = "torch_compile_corpus_v3"

CATEGORY_WEIGHTS = {
    "tensor_arithmetic": 12,
    "broadcasting": 8,
    "modules_parameters_buffers": 8,
    "inference": 6,
    "training_autograd": 8,
    "python_control_flow": 8,
    "graph_breaks_fullgraph": 8,
    "dynamic_shapes_symbolics": 8,
    "mutation_aliasing_views": 8,
    "containers_pytrees": 6,
    "decompositions": 6,
    "custom_functions": 6,
    "recompilation_guards": 4,
    "dtype_device_transitions": 4,
}

def cpu_float32_unary_abs_neg(x):
    return x.neg().abs()


def cpu_float32_unary_inputs(module):
    return (
        module.tensor(
            [[-3.25, -0.0, 1.5], [2.0, -4.5, 0.25]],
            dtype=module.float32,
        ),
    )


def cpu_float32_self_add(x):
    return x + x


def cpu_float32_self_add_method(x):
    return x.add(x)


def cpu_float32_abs_neg_reordered(x):
    return x.abs().neg()


def cpu_float32_repeated_unary_chain(x):
    return x.neg().negative().abs().absolute().neg()


def cpu_float32_add_unary_composition(x):
    y = x.neg()
    z = x.abs()
    return (y + z).add(x.negative())


def cpu_float32_self_add_inputs(module):
    return (
        module.tensor(
            [[-2.5, 0.0, 1.25], [3.0, -4.75, 6.5]],
            dtype=module.float32,
        ),
    )


def cpu_float32_scalar_inputs(module):
    return (module.tensor(-3.5, dtype=module.float32),)


def cpu_float32_empty_matrix_inputs(module):
    return (module.tensor([[], []], dtype=module.float32),)


def cpu_float32_matrix_vector_add(x, y):
    return (x + y).abs()


def cpu_float32_tensor_scalar_add(x, y):
    return x.neg().abs().add(y)


def cpu_float32_heldout_broadcast_chain(x, y):
    return (x.neg() + y.abs()).add(y)


def cpu_float32_matrix_vector_inputs(module):
    return (
        module.tensor(
            [[-2.0, 0.5, 3.0], [4.25, -5.5, 6.0]],
            dtype=module.float32,
        ),
        module.tensor([0.25, -1.5, 2.0], dtype=module.float32),
    )


def cpu_float32_tensor_scalar_inputs(module):
    return (
        module.tensor(
            [[-2.0, 0.5, 3.0], [4.25, -5.5, 6.0]],
            dtype=module.float32,
            requires_grad=True,
        ),
        module.tensor(-0.75, dtype=module.float32),
    )


def cpu_float32_heldout_broadcast_inputs(module):
    return (
        module.tensor(
            [[[-2.0, 0.5, 3.0]], [[4.25, -5.5, 6.0]]],
            dtype=module.float32,
        ),
        module.tensor([[0.25], [-1.5]], dtype=module.float32),
    )


@dataclass(frozen=True)
class CompileCorpusCase:
    name: str
    category: str
    program: object
    make_inputs: object
    fullgraph: bool = True
    dynamic: object = None
    mode: object = None
    options: object = None

    def compile_kwargs(self, backend):
        kwargs = {
            "backend": backend,
            "fullgraph": self.fullgraph,
        }
        if self.dynamic is not None:
            kwargs["dynamic"] = self.dynamic
        if self.mode is not None:
            kwargs["mode"] = self.mode
        if self.options is not None:
            kwargs["options"] = dict(self.options)
        return kwargs


COMPILE_CORPUS = (
    CompileCorpusCase(
        name="cpu_float32_unary_abs_neg",
        category="tensor_arithmetic",
        program=cpu_float32_unary_abs_neg,
        make_inputs=cpu_float32_unary_inputs,
    ),
    CompileCorpusCase(
        name="cpu_float32_self_add",
        category="tensor_arithmetic",
        program=cpu_float32_self_add,
        make_inputs=cpu_float32_self_add_inputs,
    ),
    CompileCorpusCase(
        name="cpu_float32_abs_neg_reordered",
        category="tensor_arithmetic",
        program=cpu_float32_abs_neg_reordered,
        make_inputs=cpu_float32_unary_inputs,
    ),
    CompileCorpusCase(
        name="cpu_float32_repeated_unary_chain",
        category="tensor_arithmetic",
        program=cpu_float32_repeated_unary_chain,
        make_inputs=cpu_float32_scalar_inputs,
    ),
    CompileCorpusCase(
        name="cpu_float32_add_unary_composition",
        category="tensor_arithmetic",
        program=cpu_float32_add_unary_composition,
        make_inputs=cpu_float32_empty_matrix_inputs,
    ),
    CompileCorpusCase(
        name="cpu_float32_matrix_vector_add",
        category="broadcasting",
        program=cpu_float32_matrix_vector_add,
        make_inputs=cpu_float32_matrix_vector_inputs,
    ),
    CompileCorpusCase(
        name="cpu_float32_tensor_scalar_add",
        category="broadcasting",
        program=cpu_float32_tensor_scalar_add,
        make_inputs=cpu_float32_tensor_scalar_inputs,
    ),
    CompileCorpusCase(
        name="cpu_float32_heldout_broadcast_chain",
        category="broadcasting",
        program=cpu_float32_heldout_broadcast_chain,
        make_inputs=cpu_float32_heldout_broadcast_inputs,
    ),
)


def make_recording_backend(calls):
    def backend(graph_module, example_inputs):
        calls.append((graph_module, example_inputs))
        return graph_module.forward

    return backend


def reset_reference_compile_state():
    dynamo = getattr(reference_torch, "_dynamo", None)
    if dynamo is not None:
        reset = getattr(dynamo, "reset", None)
        if reset is not None:
            reset()


class CompileCorpusMetadataTests(unittest.TestCase):
    def test_corpus_has_versioned_weighted_skeleton(self):
        self.assertEqual(COMPILE_CORPUS_VERSION, "torch_compile_corpus_v3")
        self.assertEqual(sum(CATEGORY_WEIGHTS.values()), 100)
        self.assertEqual(len(COMPILE_CORPUS), 8)

        case_names = [case.name for case in COMPILE_CORPUS]
        self.assertEqual(
            case_names,
            [
                "cpu_float32_unary_abs_neg",
                "cpu_float32_self_add",
                "cpu_float32_abs_neg_reordered",
                "cpu_float32_repeated_unary_chain",
                "cpu_float32_add_unary_composition",
                "cpu_float32_matrix_vector_add",
                "cpu_float32_tensor_scalar_add",
                "cpu_float32_heldout_broadcast_chain",
            ],
        )
        expected_categories = {
            "cpu_float32_unary_abs_neg": "tensor_arithmetic",
            "cpu_float32_self_add": "tensor_arithmetic",
            "cpu_float32_abs_neg_reordered": "tensor_arithmetic",
            "cpu_float32_repeated_unary_chain": "tensor_arithmetic",
            "cpu_float32_add_unary_composition": "tensor_arithmetic",
            "cpu_float32_matrix_vector_add": "broadcasting",
            "cpu_float32_tensor_scalar_add": "broadcasting",
            "cpu_float32_heldout_broadcast_chain": "broadcasting",
        }
        for case in COMPILE_CORPUS:
            with self.subTest(case=case.name):
                self.assertEqual(case.category, expected_categories[case.name])
                self.assertIn(case.category, CATEGORY_WEIGHTS)
                self.assertTrue(case.fullgraph)
                self.assertIsNone(case.dynamic)
                self.assertIsNone(case.mode)
                self.assertIsNone(case.options)
        self.assertNotIn("_compile_trace", torch.__all__)
        self.assertNotIn("_compile_trace_tensor_metadata", torch._C.__all__)
        self.assertNotIn("_compile_trace_grad_enabled", torch._C.__all__)
        self.assertNotIn("_compile_trace_unary", torch._C.__all__)
        self.assertNotIn("_compile_trace_binary", torch._C.__all__)
        self.assertNotIn("_compile_bytecode", torch.__all__)
        self.assertFalse(hasattr(_compile_trace, "_dis"))
        self.assertFalse(hasattr(_compile_trace, "lower_one_input_compile_graph"))


class CompileCorpusTraceTests(unittest.TestCase):
    @staticmethod
    def bytecode_instruction(opname, argval=None, argrepr="", arg=0):
        return SimpleNamespace(
            opname=opname,
            argval=argval,
            argrepr=argrepr,
            arg=arg,
        )

    def lower_with_bytecode_instructions(self, program, instructions, *inputs):
        original_get_instructions = _compile_bytecode._dis.get_instructions

        def fake_get_instructions(requested_program):
            self.assertIs(requested_program, program)
            return iter(instructions)

        try:
            _compile_bytecode._dis.get_instructions = fake_get_instructions
            return _compile_bytecode.lower_compile_graph(
                program,
                tuple(
                    _compile_trace._metadata_from_native_tensor(input)
                    for input in inputs
                ),
                name=program.__name__,
            )
        finally:
            _compile_bytecode._dis.get_instructions = original_get_instructions

    def assert_native_tensor_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertIsInstance(actual, torch.Tensor)
            self.assertEqual(
                tuple(actual.shape),
                tuple(expected.shape),
                msg=f"{case} shape mismatch",
            )
            self.assertEqual(
                actual.stride(),
                expected.stride(),
                msg=f"{case} stride mismatch",
            )
            self.assertIs(
                actual.dtype,
                expected.dtype,
                msg=f"{case} dtype mismatch",
            )
            self.assertEqual(
                actual.device,
                expected.device,
                msg=f"{case} device mismatch",
            )
            self.assertEqual(
                actual.storage_offset(),
                expected.storage_offset(),
                msg=f"{case} storage offset mismatch",
            )
            self.assertEqual(
                actual.is_contiguous(),
                expected.is_contiguous(),
                msg=f"{case} contiguity mismatch",
            )
            self.assertEqual(
                actual.requires_grad,
                expected.requires_grad,
                msg=f"{case} requires_grad mismatch",
            )
        with self.subTest(case=case, values=True):
            self.assertEqual(
                actual.tolist(),
                expected.tolist(),
                msg=(
                    f"{case} value mismatch: expected {expected.tolist()!r}, "
                    f"got {actual.tolist()!r}"
                ),
            )

    def test_bytecode_lowerer_records_general_tensor_arithmetic_graphs(self):
        cases = (
            (
                cpu_float32_unary_abs_neg,
                cpu_float32_unary_inputs,
                ["neg", "abs"],
            ),
            (
                cpu_float32_self_add,
                cpu_float32_self_add_inputs,
                ["add"],
            ),
            (
                cpu_float32_abs_neg_reordered,
                cpu_float32_unary_inputs,
                ["abs", "neg"],
            ),
            (
                cpu_float32_repeated_unary_chain,
                cpu_float32_scalar_inputs,
                ["neg", "neg", "abs", "abs", "neg"],
            ),
            (
                cpu_float32_add_unary_composition,
                cpu_float32_empty_matrix_inputs,
                ["neg", "abs", "add", "neg", "add"],
            ),
        )
        for program, make_inputs, expected_targets in cases:
            with self.subTest(program=program.__name__):
                (input,) = make_inputs(torch)
                graph = _compile_bytecode.lower_one_input_compile_graph(
                    program,
                    _compile_trace._metadata_from_native_tensor(input),
                    name=program.__name__,
                )
                expected = program(input)

                self.assertEqual(graph.name, program.__name__)
                self.assertEqual(graph.inputs[0].name, "x")
                self.assertEqual(
                    [operation.target for operation in graph.operations],
                    expected_targets,
                )
                self.assert_native_tensor_matches(
                    graph.forward(input),
                    expected,
                    case=program.__name__,
                )

    def test_bytecode_lowerer_records_two_input_broadcast_graphs(self):
        cases = (
            (
                cpu_float32_matrix_vector_add,
                cpu_float32_matrix_vector_inputs,
                ["add", "abs"],
                ("x", "y"),
            ),
            (
                cpu_float32_tensor_scalar_add,
                cpu_float32_tensor_scalar_inputs,
                ["neg", "abs", "add"],
                ("x", "y"),
            ),
            (
                cpu_float32_heldout_broadcast_chain,
                cpu_float32_heldout_broadcast_inputs,
                ["neg", "abs", "add", "add"],
                ("x", "y"),
            ),
        )
        for program, make_inputs, expected_targets, input_names in cases:
            with self.subTest(program=program.__name__):
                inputs = make_inputs(torch)
                graph = _compile_bytecode.lower_two_input_compile_graph(
                    program,
                    *(
                        _compile_trace._metadata_from_native_tensor(input)
                        for input in inputs
                    ),
                    name=program.__name__,
                )
                expected = program(*inputs)

                self.assertEqual(graph.name, program.__name__)
                self.assertEqual(
                    tuple(input.name for input in graph.inputs),
                    input_names,
                )
                self.assertEqual(
                    [operation.target for operation in graph.operations],
                    expected_targets,
                )
                self.assertEqual(graph.output_metadata.shape, tuple(expected.shape))
                self.assertEqual(graph.output_metadata.stride, expected.stride())
                self.assert_native_tensor_matches(
                    graph.forward(*inputs),
                    expected,
                    case=program.__name__,
                )

    def test_bytecode_lowerer_accepts_cpython_314_borrowed_local_loads(self):
        def program(x):
            raise AssertionError("synthetic bytecode test must not run")

        input = torch.tensor([[-3.0, 0.0, 4.5]], dtype=torch.float32)
        instructions = (
            self.bytecode_instruction("RESUME"),
            self.bytecode_instruction("LOAD_FAST_BORROW", "x", "x"),
            self.bytecode_instruction("LOAD_ATTR", "neg", "NULL|self + neg", 1),
            self.bytecode_instruction("CALL", arg=0),
            self.bytecode_instruction("LOAD_ATTR", "abs", "NULL|self + abs", 1),
            self.bytecode_instruction("CALL", arg=0),
            self.bytecode_instruction("RETURN_VALUE"),
        )

        graph = self.lower_with_bytecode_instructions(program, instructions, input)

        self.assertEqual(
            [operation.target for operation in graph.operations],
            ["neg", "abs"],
        )
        self.assert_native_tensor_matches(
            graph.forward(input),
            input.neg().abs(),
            case="LOAD_FAST_BORROW",
        )

    def test_bytecode_lowerer_rejects_cpython_310_keyword_method_call(self):
        def program(x):
            raise AssertionError("synthetic bytecode test must not run")

        input = torch.tensor([-2.0, 0.5, 3.0], dtype=torch.float32)
        instructions = (
            self.bytecode_instruction("LOAD_FAST", "x", "x"),
            self.bytecode_instruction("LOAD_ATTR", "add", "add", 0),
            self.bytecode_instruction("LOAD_FAST", "x", "x"),
            self.bytecode_instruction("LOAD_CONST", ("other",), "('other',)", 1),
            self.bytecode_instruction("CALL_FUNCTION_KW", arg=1),
            self.bytecode_instruction("RETURN_VALUE"),
        )

        with self.assertRaisesRegex(
            _compile_trace.CompileTraceUnsupportedError,
            "keyword arguments",
        ):
            self.lower_with_bytecode_instructions(program, instructions, input)

    def test_bytecode_lowerer_rejects_cpython_313_keyword_method_call(self):
        def program(x):
            raise AssertionError("synthetic bytecode test must not run")

        input = torch.tensor([-2.0, 0.5, 3.0], dtype=torch.float32)
        instructions = (
            self.bytecode_instruction("LOAD_FAST", "x", "x"),
            self.bytecode_instruction("LOAD_ATTR", "add", "NULL|self + add", 1),
            self.bytecode_instruction("LOAD_FAST", "x", "x"),
            self.bytecode_instruction("LOAD_CONST", ("other",), "('other',)", 1),
            self.bytecode_instruction("CALL_KW", arg=1),
            self.bytecode_instruction("RETURN_VALUE"),
        )

        with self.assertRaisesRegex(
            _compile_trace.CompileTraceUnsupportedError,
            "keyword arguments",
        ):
            self.lower_with_bytecode_instructions(program, instructions, input)

    def test_bytecode_lowerer_rejects_cpython_313_to_bool_as_control_flow(self):
        def program(x):
            raise AssertionError("synthetic bytecode test must not run")

        input = torch.tensor([-2.0, 0.5, 3.0], dtype=torch.float32)
        instructions = (
            self.bytecode_instruction("LOAD_FAST", "x", "x"),
            self.bytecode_instruction("TO_BOOL"),
            self.bytecode_instruction("POP_JUMP_IF_FALSE"),
            self.bytecode_instruction("LOAD_FAST", "x", "x"),
            self.bytecode_instruction("RETURN_VALUE"),
        )

        with self.assertRaisesRegex(
            _compile_trace.CompileTraceUnsupportedError,
            "control flow",
        ):
            self.lower_with_bytecode_instructions(program, instructions, input)

    def test_bytecode_lowerer_accepts_combined_local_load_opcodes(self):
        def program(x):
            raise AssertionError("synthetic bytecode test must not run")

        input = torch.tensor([-2.0, 0.5, 3.0], dtype=torch.float32)
        variants = (
            (
                "LOAD_FAST_LOAD_FAST",
                ("x", "x"),
                "",
            ),
            (
                "LOAD_FAST_BORROW_LOAD_FAST_BORROW",
                None,
                "x, x",
            ),
            (
                "LOAD_FAST_BORROW_LOAD_FAST",
                None,
                "(x, x)",
            ),
        )
        for opname, argval, argrepr in variants:
            with self.subTest(opname=opname):
                instructions = (
                    self.bytecode_instruction(opname, argval, argrepr),
                    self.bytecode_instruction("BINARY_OP", argrepr="+"),
                    self.bytecode_instruction("RETURN_VALUE"),
                )

                graph = self.lower_with_bytecode_instructions(
                    program,
                    instructions,
                    input,
                )

                self.assertEqual(
                    [operation.target for operation in graph.operations],
                    ["add"],
                )
                self.assertEqual(graph.operations[0].inputs, ("x", "x"))
                self.assert_native_tensor_matches(
                    graph.forward(input),
                    input + input,
                    case=opname,
                )

    def test_bytecode_lowerer_accepts_combined_store_then_load_opcode(self):
        def program(x):
            y = x
            z = y
            return z

        input = torch.tensor([[-2.0, 0.5], [3.0, -4.0]], dtype=torch.float32)
        instructions = (
            self.bytecode_instruction("RESUME"),
            self.bytecode_instruction("LOAD_FAST_BORROW", "x", "x"),
            self.bytecode_instruction("UNARY_NEGATIVE"),
            self.bytecode_instruction("STORE_FAST_LOAD_FAST", ("y", "y")),
            self.bytecode_instruction("LOAD_FAST_BORROW", "x", "x"),
            self.bytecode_instruction("BINARY_OP", argrepr="+"),
            self.bytecode_instruction("STORE_FAST_LOAD_FAST", None, "z, z"),
            self.bytecode_instruction("LOAD_METHOD", "abs", "abs"),
            self.bytecode_instruction("CALL", arg=0),
            self.bytecode_instruction("RETURN_VALUE"),
        )

        graph = self.lower_with_bytecode_instructions(program, instructions, input)

        self.assertEqual(
            [operation.target for operation in graph.operations],
            ["neg", "add", "abs"],
        )
        self.assertEqual(graph.operations[1].inputs, ("neg_0", "x"))
        self.assert_native_tensor_matches(
            graph.forward(input),
            (input.neg() + input).abs(),
            case="STORE_FAST_LOAD_FAST",
        )

    def test_unary_abs_neg_records_private_immutable_graph(self):
        graph = _compile_trace.trace_one_input_compile_graph(
            cpu_float32_unary_abs_neg,
            cpu_float32_unary_inputs,
            name="cpu_float32_unary_abs_neg",
        )

        self.assertEqual(graph.name, "cpu_float32_unary_abs_neg")
        self.assertEqual(graph.output, "abs_1")
        self.assertEqual(len(graph.inputs), 1)
        self.assertEqual(len(graph.operations), 2)

        input_metadata = graph.inputs[0].metadata
        self.assertEqual(graph.inputs[0].name, "arg0")
        self.assertEqual(graph.inputs[0].index, 0)
        self.assertEqual(input_metadata.shape, (2, 3))
        self.assertEqual(input_metadata.stride, (3, 1))
        self.assertIs(input_metadata.dtype, _compile_trace.float32)
        self.assertEqual(input_metadata.device, "cpu")
        self.assertFalse(input_metadata.requires_grad)

        neg, abs_op = graph.operations
        self.assertEqual(neg.name, "neg_0")
        self.assertEqual(neg.op, "call_method")
        self.assertEqual(neg.target, "neg")
        self.assertEqual(neg.inputs, ("arg0",))
        self.assertEqual(neg.metadata, input_metadata)

        self.assertEqual(abs_op.name, "abs_1")
        self.assertEqual(abs_op.op, "call_method")
        self.assertEqual(abs_op.target, "abs")
        self.assertEqual(abs_op.inputs, ("neg_0",))
        self.assertEqual(abs_op.metadata, input_metadata)
        self.assertEqual(graph.output_metadata, input_metadata)

        with self.assertRaises(FrozenInstanceError):
            graph.output = "changed"
        with self.assertRaises(AttributeError):
            graph.operations.append(abs_op)

    def test_binary_self_add_records_private_immutable_graph(self):
        for program, expected_name in (
            (cpu_float32_self_add, "cpu_float32_self_add"),
            (cpu_float32_self_add_method, "cpu_float32_self_add_method"),
        ):
            with self.subTest(program=expected_name):
                graph = _compile_trace.trace_one_input_compile_graph(
                    program,
                    cpu_float32_self_add_inputs,
                    name=expected_name,
                )

                self.assertEqual(graph.name, expected_name)
                self.assertEqual(graph.output, "add_0")
                self.assertEqual(len(graph.inputs), 1)
                self.assertEqual(len(graph.operations), 1)

                input_metadata = graph.inputs[0].metadata
                self.assertEqual(graph.inputs[0].name, "arg0")
                self.assertEqual(graph.inputs[0].index, 0)
                self.assertEqual(input_metadata.shape, (2, 3))
                self.assertEqual(input_metadata.stride, (3, 1))
                self.assertIs(input_metadata.dtype, _compile_trace.float32)
                self.assertEqual(input_metadata.device, "cpu")
                self.assertFalse(input_metadata.requires_grad)

                (add_op,) = graph.operations
                self.assertEqual(add_op.name, "add_0")
                self.assertEqual(add_op.op, "call_method")
                self.assertEqual(add_op.target, "add")
                self.assertEqual(add_op.inputs, ("arg0", "arg0"))
                self.assertEqual(add_op.metadata, input_metadata)
                self.assertEqual(graph.output_metadata, input_metadata)

                with self.assertRaises(FrozenInstanceError):
                    add_op.target = "sub"
                with self.assertRaises(FrozenInstanceError):
                    add_op.metadata = None
                with self.assertRaises(AttributeError):
                    graph.operations.append(add_op)

    def test_binary_broadcast_records_private_immutable_graph(self):
        graph = _compile_trace.trace_two_input_compile_graph(
            cpu_float32_matrix_vector_add,
            cpu_float32_matrix_vector_inputs,
            name="cpu_float32_matrix_vector_add",
        )
        inputs = cpu_float32_matrix_vector_inputs(torch)
        expected = cpu_float32_matrix_vector_add(*inputs)

        self.assertEqual(graph.name, "cpu_float32_matrix_vector_add")
        self.assertEqual(graph.output, "abs_1")
        self.assertEqual(len(graph.inputs), 2)
        self.assertEqual(
            tuple(input.name for input in graph.inputs),
            ("arg0", "arg1"),
        )
        self.assertEqual(tuple(input.index for input in graph.inputs), (0, 1))

        left_metadata = graph.inputs[0].metadata
        right_metadata = graph.inputs[1].metadata
        self.assertEqual(left_metadata.shape, (2, 3))
        self.assertEqual(left_metadata.stride, (3, 1))
        self.assertEqual(right_metadata.shape, (3,))
        self.assertEqual(right_metadata.stride, (1,))
        self.assertIs(left_metadata.dtype, _compile_trace.float32)
        self.assertIs(right_metadata.dtype, _compile_trace.float32)

        add_op, abs_op = graph.operations
        self.assertEqual(add_op.name, "add_0")
        self.assertEqual(add_op.inputs, ("arg0", "arg1"))
        self.assertEqual(add_op.metadata.shape, tuple(expected.shape))
        self.assertEqual(add_op.metadata.stride, expected.stride())
        self.assertFalse(add_op.metadata.requires_grad)
        self.assertEqual(abs_op.name, "abs_1")
        self.assertEqual(abs_op.inputs, ("add_0",))
        self.assertEqual(abs_op.metadata, add_op.metadata)
        self.assertEqual(graph.output_metadata, add_op.metadata)

        with self.assertRaises(FrozenInstanceError):
            graph.inputs = ()
        with self.assertRaises(FrozenInstanceError):
            add_op.inputs = ()

    def test_unary_abs_neg_executes_private_native_graph(self):
        case = COMPILE_CORPUS[0]
        graph = _compile_trace.trace_one_input_compile_graph(
            case.program,
            case.make_inputs,
            name=case.name,
        )
        inputs = case.make_inputs(torch)
        expected = case.program(*inputs)

        self.assert_native_tensor_matches(
            graph.forward(*inputs),
            expected,
            case=case.name,
        )
        self.assert_native_tensor_matches(
            _compile_trace.execute_compile_trace_graph(graph, *inputs),
            expected,
            case=f"{case.name} function executor",
        )

    def test_binary_self_add_executes_private_native_graph_for_key_layouts(self):
        cases = (
            ("scalar operator", torch.tensor(2.5, dtype=torch.float32), False),
            ("scalar method", torch.tensor(-0.0, dtype=torch.float32), True),
            ("empty operator", torch.tensor([], dtype=torch.float32), False),
            (
                "contiguous method",
                torch.tensor(
                    [[-3.25, -0.0, 1.5], [2.0, -4.5, 0.25]],
                    dtype=torch.float32,
                ),
                True,
            ),
            (
                "offset noncontiguous operator",
                torch.tensor(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    dtype=torch.float32,
                ).transpose(0, 1)[1],
                False,
            ),
        )
        for case, input, use_method in cases:
            with self.subTest(case=case):
                if case.startswith("offset"):
                    self.assertGreater(input.storage_offset(), 0)
                    self.assertFalse(input.is_contiguous())

                recorder = _compile_trace.CompileTraceRecorder(name=case)
                proxy = recorder.input(
                    shape=tuple(input.shape),
                    stride=input.stride(),
                    dtype=_compile_trace.float32,
                    device="cpu",
                    requires_grad=input.requires_grad,
                )
                output_proxy = proxy.add(proxy) if use_method else proxy + proxy
                graph = recorder.finish(output_proxy)
                expected = input.add(input) if use_method else input + input

                self.assertEqual(graph.operations[0].target, "add")
                self.assertEqual(graph.operations[0].inputs, ("arg0", "arg0"))
                self.assertEqual(
                    graph.operations[0].metadata.shape,
                    tuple(expected.shape),
                )
                self.assertEqual(
                    graph.operations[0].metadata.stride,
                    expected.stride(),
                )
                self.assert_native_tensor_matches(
                    graph.forward(input),
                    expected,
                    case=case,
                )
                self.assert_native_tensor_matches(
                    _compile_trace.execute_compile_trace_graph(graph, input),
                    expected,
                    case=f"{case} function executor",
                )

    def test_binary_broadcast_executes_private_native_graph_for_key_layouts(self):
        cases = (
            (
                "matrix vector",
                torch.tensor(
                    [[-2.0, 0.5, 3.0], [4.25, -5.5, 6.0]],
                    dtype=torch.float32,
                ),
                torch.tensor([0.25, -1.5, 2.0], dtype=torch.float32),
            ),
            (
                "tensor scalar",
                torch.tensor(
                    [[-2.0, 0.5, 3.0], [4.25, -5.5, 6.0]],
                    dtype=torch.float32,
                ),
                torch.tensor(-0.75, dtype=torch.float32),
            ),
            (
                "scalar tensor",
                torch.tensor(1.25, dtype=torch.float32),
                torch.tensor(
                    [[-2.0, 0.5, 3.0], [4.25, -5.5, 6.0]],
                    dtype=torch.float32,
                ),
            ),
            (
                "empty matrix vector",
                torch.zeros((0, 3), dtype=torch.float32),
                torch.tensor([0.25, -1.5, 2.0], dtype=torch.float32),
            ),
        )
        for case, left_input, right_input in cases:
            with self.subTest(case=case):
                recorder = _compile_trace.CompileTraceRecorder(name=case)
                left_proxy = recorder.input(
                    shape=tuple(left_input.shape),
                    stride=left_input.stride(),
                    dtype=_compile_trace.float32,
                    device="cpu",
                    requires_grad=left_input.requires_grad,
                )
                right_proxy = recorder.input(
                    shape=tuple(right_input.shape),
                    stride=right_input.stride(),
                    dtype=_compile_trace.float32,
                    device="cpu",
                    requires_grad=right_input.requires_grad,
                )
                graph = recorder.finish(left_proxy + right_proxy)
                expected = left_input + right_input

                self.assertEqual(graph.operations[0].inputs, ("arg0", "arg1"))
                self.assertEqual(
                    graph.operations[0].metadata.shape,
                    tuple(expected.shape),
                )
                self.assertEqual(
                    graph.operations[0].metadata.stride,
                    expected.stride(),
                )
                self.assert_native_tensor_matches(
                    graph.forward(left_input, right_input),
                    expected,
                    case=case,
                )
                self.assert_native_tensor_matches(
                    _compile_trace.execute_compile_trace_graph(
                        graph,
                        left_input,
                        right_input,
                    ),
                    expected,
                    case=f"{case} function executor",
                )

    def test_unary_abs_neg_executor_matches_no_grad_requires_grad_outputs(self):
        def make_inputs(module):
            return (
                module.tensor(
                    [[-1.0, 2.0]],
                    dtype=module.float32,
                    requires_grad=True,
                ),
            )

        graph = _compile_trace.trace_one_input_compile_graph(
            cpu_float32_unary_abs_neg,
            make_inputs,
            name="cpu_float32_unary_abs_neg_requires_grad",
        )
        input = make_inputs(torch)[0]
        expected_with_grad = cpu_float32_unary_abs_neg(input)

        self.assertTrue(graph.inputs[0].metadata.requires_grad)
        self.assertTrue(graph.operations[0].metadata.requires_grad)
        self.assertTrue(graph.output_metadata.requires_grad)
        self.assert_native_tensor_matches(
            graph.forward(input),
            expected_with_grad,
            case="requires_grad grad-enabled",
        )

        with torch.no_grad():
            expected_no_grad = cpu_float32_unary_abs_neg(input)
            actual_no_grad = graph.forward(input)

        self.assertFalse(expected_no_grad.requires_grad)
        self.assert_native_tensor_matches(
            actual_no_grad,
            expected_no_grad,
            case="requires_grad no_grad",
        )

    def test_binary_self_add_executor_matches_no_grad_requires_grad_outputs(self):
        def make_inputs(module):
            return (
                module.tensor(
                    [[-1.0, 2.0]],
                    dtype=module.float32,
                    requires_grad=True,
                ),
            )

        graph = _compile_trace.trace_one_input_compile_graph(
            cpu_float32_self_add,
            make_inputs,
            name="cpu_float32_self_add_requires_grad",
        )
        input = make_inputs(torch)[0]
        expected_with_grad = cpu_float32_self_add(input)

        self.assertTrue(graph.inputs[0].metadata.requires_grad)
        self.assertTrue(graph.operations[0].metadata.requires_grad)
        self.assertTrue(graph.output_metadata.requires_grad)
        self.assert_native_tensor_matches(
            graph.forward(input),
            expected_with_grad,
            case="binary requires_grad grad-enabled",
        )

        with torch.no_grad():
            expected_no_grad = cpu_float32_self_add(input)
            actual_no_grad = graph.forward(input)

        self.assertFalse(expected_no_grad.requires_grad)
        self.assert_native_tensor_matches(
            actual_no_grad,
            expected_no_grad,
            case="binary requires_grad no_grad",
        )

    def test_binary_broadcast_executor_matches_no_grad_requires_grad_outputs(self):
        def make_inputs(module):
            return (
                module.tensor(
                    [[-1.0, 2.0]],
                    dtype=module.float32,
                    requires_grad=False,
                ),
                module.tensor(
                    [1.5, -0.5],
                    dtype=module.float32,
                    requires_grad=True,
                ),
            )

        graph = _compile_trace.trace_two_input_compile_graph(
            cpu_float32_matrix_vector_add,
            make_inputs,
            name="cpu_float32_matrix_vector_add_requires_grad",
        )
        inputs = make_inputs(torch)
        expected_with_grad = cpu_float32_matrix_vector_add(*inputs)

        self.assertFalse(graph.inputs[0].metadata.requires_grad)
        self.assertTrue(graph.inputs[1].metadata.requires_grad)
        self.assertTrue(graph.operations[0].metadata.requires_grad)
        self.assertTrue(graph.output_metadata.requires_grad)
        self.assert_native_tensor_matches(
            graph.forward(*inputs),
            expected_with_grad,
            case="broadcast requires_grad grad-enabled",
        )

        with torch.no_grad():
            expected_no_grad = cpu_float32_matrix_vector_add(*inputs)
            actual_no_grad = graph.forward(*inputs)

        self.assertFalse(expected_no_grad.requires_grad)
        self.assert_native_tensor_matches(
            actual_no_grad,
            expected_no_grad,
            case="broadcast requires_grad no_grad",
        )

    def test_unary_output_metadata_matches_native_stride_planning(self):
        cases = (
            (
                "singleton dimension",
                torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32).t(),
            ),
            (
                "dense transpose",
                torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32).t(),
            ),
            (
                "empty transpose",
                torch.zeros((2, 0, 3), dtype=torch.float32).transpose(0, 2)[1],
            ),
            (
                "channels last",
                torch.zeros((2, 3, 4, 5), dtype=torch.float32).contiguous(
                    memory_format=torch.channels_last
                ),
            ),
            (
                "channels last 3d",
                torch.zeros((2, 3, 4, 5, 6), dtype=torch.float32).contiguous(
                    memory_format=torch.channels_last_3d
                ),
            ),
            (
                "channels last 3d singleton transpose",
                torch.zeros(
                    (2, 3, 1, 5, 6),
                    dtype=torch.float32,
                )
                .contiguous(memory_format=torch.channels_last_3d)
                .transpose(0, 2),
            ),
        )
        for case, input in cases:
            with self.subTest(case=case):
                recorder = _compile_trace.CompileTraceRecorder()
                proxy = recorder.input(
                    shape=tuple(input.shape),
                    stride=input.stride(),
                    dtype=_compile_trace.float32,
                    device="cpu",
                    requires_grad=input.requires_grad,
                )
                neg_proxy = proxy.neg()
                output_proxy = neg_proxy.abs()
                graph = recorder.finish(output_proxy)

                expected_neg = input.neg()
                expected = expected_neg.abs()
                self.assertEqual(
                    graph.operations[0].metadata.stride,
                    expected_neg.stride(),
                )
                self.assertEqual(
                    graph.operations[1].metadata.stride,
                    expected.stride(),
                )
                self.assertEqual(graph.output_metadata.stride, expected.stride())
                self.assert_native_tensor_matches(
                    graph.forward(input),
                    expected,
                    case=case,
                )

    def test_private_executor_bypasses_active_torch_function_mode(self):
        from torch_rs.overrides import TorchFunctionMode

        graph = _compile_trace.trace_one_input_compile_graph(
            cpu_float32_unary_abs_neg,
            lambda module: (
                module.tensor([[-1.0, 2.0]], dtype=module.float32),
            ),
            name="cpu_float32_unary_abs_neg",
        )
        input = torch.tensor([[-1.0, 2.0]], dtype=torch.float32)
        expected = cpu_float32_unary_abs_neg(input)
        mode_calls = []

        class ReplacingMode(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                mode_calls.append(getattr(func, "__name__", repr(func)))
                if getattr(func, "__name__", None) == "abs":
                    return torch.tensor([[99.0, 100.0]], dtype=torch.float32)
                raise AssertionError(
                    "private compile trace execution dispatched through "
                    f"TorchFunctionMode for {mode_calls[-1]}"
                )

        with ReplacingMode():
            actual = graph.forward(input)

        self.assertEqual(mode_calls, [])
        self.assert_native_tensor_matches(
            actual,
            expected,
            case="active TorchFunctionMode",
        )

    def test_binary_private_executor_bypasses_active_torch_function_mode(self):
        from torch_rs.overrides import TorchFunctionMode

        graph = _compile_trace.trace_one_input_compile_graph(
            cpu_float32_self_add,
            lambda module: (
                module.tensor([[-1.0, 2.0]], dtype=module.float32),
            ),
            name="cpu_float32_self_add",
        )
        input = torch.tensor([[-1.0, 2.0]], dtype=torch.float32)
        expected = cpu_float32_self_add(input)
        mode_calls = []

        class ReplacingMode(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                mode_calls.append(getattr(func, "__name__", repr(func)))
                if getattr(func, "__name__", None) == "add":
                    return torch.tensor([[99.0, 100.0]], dtype=torch.float32)
                raise AssertionError(
                    "private compile trace execution dispatched through "
                    f"TorchFunctionMode for {mode_calls[-1]}"
                )

        with ReplacingMode():
            actual = graph.forward(input)

        self.assertEqual(mode_calls, [])
        self.assert_native_tensor_matches(
            actual,
            expected,
            case="binary active TorchFunctionMode",
        )

    def test_private_executor_rejects_runtime_metadata_mismatch_clearly(self):
        graph = _compile_trace.trace_one_input_compile_graph(
            cpu_float32_unary_abs_neg,
            cpu_float32_unary_inputs,
            name="cpu_float32_unary_abs_neg",
        )
        mismatched = torch.tensor([1.0], dtype=torch.float32)

        with self.assertRaisesRegex(
            ValueError,
            (
                "metadata mismatch for 'arg0': "
                r"shape expected \(2, 3\), got \(1,\)"
            ),
        ):
            graph.forward(mismatched)

    def test_private_executor_rejects_second_input_metadata_mismatch_clearly(self):
        graph = _compile_trace.trace_two_input_compile_graph(
            cpu_float32_matrix_vector_add,
            cpu_float32_matrix_vector_inputs,
            name="cpu_float32_matrix_vector_add",
        )
        left, _ = cpu_float32_matrix_vector_inputs(torch)
        mismatched = torch.tensor(-0.5, dtype=torch.float32)

        with self.assertRaisesRegex(
            ValueError,
            (
                "metadata mismatch for 'arg1': "
                r"shape expected \(3,\), got \(\)"
            ),
        ):
            graph.forward(left, mismatched)

    def test_proxy_unsupported_operations_fail_clearly(self):
        recorder = _compile_trace.CompileTraceRecorder()
        x = recorder.input(shape=(2, 3))

        def augmented_add():
            value = x
            value += x
            return value

        for operation, call in (
            ("Tensor.__iadd__", augmented_add),
            ("Tensor.__sub__", lambda: x - x),
            ("Tensor.relu", lambda: x.relu()),
            ("Tensor.__bool__", lambda: bool(x)),
            ("Tensor.positive", lambda: x.positive()),
        ):
            with self.subTest(operation=operation):
                with self.assertRaises(
                    _compile_trace.CompileTraceUnsupportedError
                ) as raised:
                    call()
                message = str(raised.exception)
                self.assertIn(operation, message)
                self.assertIn("Tensor.neg", message)
                self.assertIn("Tensor.abs", message)
                self.assertIn("Tensor.add", message)

    def test_augmented_self_add_aliasing_rejects_instead_of_recording_add(self):
        def augmented_alias_program(x):
            y = x
            x += x
            return y

        with self.assertRaisesRegex(
            _compile_trace.CompileTraceUnsupportedError,
            "Tensor.__iadd__",
        ):
            _compile_trace.trace_one_input_compile_graph(
                augmented_alias_program,
                cpu_float32_self_add_inputs,
                name="cpu_float32_augmented_self_add",
            )

    def test_binary_proxy_rejects_non_tensor_or_mixed_recorder_operands_clearly(self):
        recorder = _compile_trace.CompileTraceRecorder()
        other_recorder = _compile_trace.CompileTraceRecorder()
        x = recorder.input(shape=(2, 3))
        y = other_recorder.input(shape=(2, 3))

        for operation, call, expected in (
            (
                "Tensor.__add__",
                lambda: x + 1,
                r"Tensor\.__add__ only supports Tensor operands, got int for right operand",
            ),
            (
                "Tensor.__radd__",
                lambda: 1 + x,
                r"Tensor\.__radd__ only supports Tensor operands, got int",
            ),
            (
                "Tensor.add",
                lambda: x.add("value"),
                r"Tensor\.add only supports Tensor operands, got str for right operand",
            ),
            (
                "Tensor.__add__ mixed recorder",
                lambda: x + y,
                "cannot mix Tensor operands from different recorders",
            ),
            (
                "Tensor.add mixed recorder",
                lambda: x.add(y),
                "cannot mix Tensor operands from different recorders",
            ),
        ):
            with self.subTest(operation=operation):
                with self.assertRaisesRegex(
                    _compile_trace.CompileTraceUnsupportedError,
                    expected,
                ):
                    call()

    def test_binary_proxy_rejects_unbroadcastable_operands_clearly(self):
        recorder = _compile_trace.CompileTraceRecorder()
        x = recorder.input(shape=(2, 3))
        y = recorder.input(shape=(2,))

        with self.assertRaisesRegex(
            _compile_trace.CompileTraceUnsupportedError,
            r"not broadcastable: \(2, 3\) and \(2,\)",
        ):
            x + y

    def test_operation_names_skip_existing_input_names(self):
        recorder = _compile_trace.CompileTraceRecorder()
        x = recorder.input(name="neg_0", shape=(2,))
        first = x.neg()
        second = first.neg()
        graph = recorder.finish(second)

        value_names = (
            *(input.name for input in graph.inputs),
            *(operation.name for operation in graph.operations),
        )
        self.assertEqual(len(value_names), len(set(value_names)))
        self.assertEqual(graph.inputs[0].name, "neg_0")
        self.assertEqual(
            [(operation.name, operation.inputs) for operation in graph.operations],
            [
                ("neg_1", ("neg_0",)),
                ("neg_2", ("neg_1",)),
            ],
        )
        self.assertEqual(graph.output, "neg_2")

    def test_empty_inner_dimension_strides_match_native_tensor_constructor(self):
        recorder = _compile_trace.CompileTraceRecorder()
        trace_module = _compile_trace.CompileTraceTorchModule(recorder)
        proxy = trace_module.tensor([[], []], dtype=trace_module.float32)
        native = torch.tensor([[], []], dtype=torch.float32)

        self.assertEqual(proxy.shape, tuple(native.shape))
        self.assertEqual(proxy.stride(), native.stride())
        self.assertEqual(proxy.shape, (2, 0))
        self.assertEqual(proxy.stride(), (1, 1))

    def test_private_trace_does_not_import_pytorch_or_invoke_backend(self):
        script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch
from torch_rs import _compile_trace

def program(x):
    return x.neg().abs()

def self_add(x):
    return x + x

def matrix_vector(x, y):
    return (x + y).abs()

def make_inputs(module):
    return (
        module.tensor(
            [[-3.25, -0.0, 1.5], [2.0, -4.5, 0.25]],
            dtype=module.float32,
        ),
    )

def make_broadcast_inputs(module):
    return (
        module.tensor(
            [[-2.0, 0.5, 3.0], [4.25, -5.5, 6.0]],
            dtype=module.float32,
        ),
        module.tensor([0.25, -1.5, 2.0], dtype=module.float32),
    )

backend_calls = []

def backend(graph_module, example_inputs):
    backend_calls.append((graph_module, example_inputs))
    return graph_module.forward

graph = _compile_trace.trace_one_input_compile_graph(
    program,
    make_inputs,
    name="cpu_float32_unary_abs_neg",
)
assert graph.output == "abs_1"
assert [operation.target for operation in graph.operations] == ["neg", "abs"]
native_input = make_inputs(torch)[0]
expected = program(native_input)
for actual in (
    graph.forward(native_input),
    _compile_trace.execute_compile_trace_graph(graph, native_input),
):
    assert actual.tolist() == expected.tolist()
    assert actual.shape == expected.shape
    assert actual.stride() == expected.stride()
    assert actual.dtype is expected.dtype
    assert actual.device == expected.device
    assert actual.requires_grad is expected.requires_grad
assert backend_calls == []
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)

self_add_graph = _compile_trace.trace_one_input_compile_graph(
    self_add,
    make_inputs,
    name="cpu_float32_self_add",
)
assert self_add_graph.output == "add_0"
assert [operation.target for operation in self_add_graph.operations] == ["add"]
assert self_add_graph.operations[0].inputs == ("arg0", "arg0")
self_add_expected = self_add(native_input)
for actual in (
    self_add_graph.forward(native_input),
    _compile_trace.execute_compile_trace_graph(self_add_graph, native_input),
):
    assert actual.tolist() == self_add_expected.tolist()
    assert actual.shape == self_add_expected.shape
    assert actual.stride() == self_add_expected.stride()
    assert actual.dtype is self_add_expected.dtype
    assert actual.device == self_add_expected.device
    assert actual.requires_grad is self_add_expected.requires_grad
assert backend_calls == []
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)

matrix_vector_graph = _compile_trace.trace_two_input_compile_graph(
    matrix_vector,
    make_broadcast_inputs,
    name="cpu_float32_matrix_vector_add",
)
assert matrix_vector_graph.output == "abs_1"
assert [operation.target for operation in matrix_vector_graph.operations] == [
    "add",
    "abs",
]
assert matrix_vector_graph.operations[0].inputs == ("arg0", "arg1")
native_broadcast_inputs = make_broadcast_inputs(torch)
matrix_vector_expected = matrix_vector(*native_broadcast_inputs)
for actual in (
    matrix_vector_graph.forward(*native_broadcast_inputs),
    _compile_trace.execute_compile_trace_graph(
        matrix_vector_graph,
        *native_broadcast_inputs,
    ),
):
    assert actual.tolist() == matrix_vector_expected.tolist()
    assert actual.shape == matrix_vector_expected.shape
    assert actual.stride() == matrix_vector_expected.stride()
    assert actual.dtype is matrix_vector_expected.dtype
    assert actual.device == matrix_vector_expected.device
    assert actual.requires_grad is matrix_vector_expected.requires_grad
assert backend_calls == []
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)

compiled = torch.compile(self_add, backend="eager", fullgraph=True)
compiled_actual = compiled(native_input)
assert compiled_actual.tolist() == self_add_expected.tolist()
assert compiled_actual.shape == self_add_expected.shape
assert compiled_actual.stride() == self_add_expected.stride()
assert compiled_actual.dtype is self_add_expected.dtype
assert compiled_actual.device == self_add_expected.device
assert compiled_actual.requires_grad is self_add_expected.requires_grad
assert backend_calls == []
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)

compiled_broadcast = torch.compile(matrix_vector, backend="eager", fullgraph=True)
compiled_broadcast_actual = compiled_broadcast(*native_broadcast_inputs)
assert compiled_broadcast_actual.tolist() == matrix_vector_expected.tolist()
assert compiled_broadcast_actual.shape == matrix_vector_expected.shape
assert compiled_broadcast_actual.stride() == matrix_vector_expected.stride()
assert compiled_broadcast_actual.dtype is matrix_vector_expected.dtype
assert compiled_broadcast_actual.device == matrix_vector_expected.device
assert compiled_broadcast_actual.requires_grad is matrix_vector_expected.requires_grad
assert backend_calls == []
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)

compiled_with_callable_backend = torch.compile(self_add, backend=backend)
try:
    compiled_with_callable_backend(make_inputs(torch)[0])
except NotImplementedError as error:
    assert str(error) == (
        "torch.compile(): only backend='eager', fullgraph=True straight-line "
        "Tensor neg/abs/add functions with one or two positional exact native CPU "
        "float32 Tensor are supported; eager fallback, installed-PyTorch "
        "forwarding, callable backend invocation, CUDA compilation, and broader "
        "graph capture remain unsupported"
    )
else:
    raise AssertionError("callable backend compile should remain non-executing")
assert backend_calls == []
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TorchCompileCorpusReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != REFERENCE_PYTORCH_VERSION:
            raise AssertionError(
                "torch.compile corpus eligibility requires pinned PyTorch "
                f"{REFERENCE_PYTORCH_VERSION}"
            )

    def assert_reference_eligible(self, case):
        reset_reference_compile_state()
        self.addCleanup(reset_reference_compile_state)
        backend_calls = []
        backend = make_recording_backend(backend_calls)
        inputs = case.make_inputs(reference_torch)
        expected = case.program(*case.make_inputs(reference_torch))

        for index, tensor in enumerate(inputs):
            with self.subTest(case=case.name, input=index):
                self.assertEqual(tensor.dtype, reference_torch.float32)
                self.assertEqual(tensor.device.type, "cpu")

        compiled = reference_torch.compile(
            case.program,
            **case.compile_kwargs(backend),
        )
        actual = compiled(*inputs)
        reference_torch.testing.assert_close(actual, expected)
        self.assertGreaterEqual(len(backend_calls), 1)

    def test_reference_pytorch_2_13_accepts_all_eligible_cases(self):
        for case in COMPILE_CORPUS:
            with self.subTest(case=case.name):
                self.assert_reference_eligible(case)

    def test_torch_rs_compile_runs_eligible_eager_cases_natively(self):
        for case in COMPILE_CORPUS:
            with self.subTest(case=case.name):
                self.assert_reference_eligible(case)

                inputs = case.make_inputs(torch)
                expected = case.program(*case.make_inputs(torch))
                compiled = torch.compile(
                    case.program,
                    **case.compile_kwargs("eager"),
                )
                actual = compiled(*inputs)
                self.assertIs(compiled._torch_rs_compile_backend, "eager")
                self.assertEqual(actual.tolist(), expected.tolist())
                self.assertEqual(tuple(actual.shape), tuple(expected.shape))
                self.assertEqual(actual.stride(), expected.stride())
                self.assertIs(actual.dtype, expected.dtype)
                self.assertEqual(actual.device, expected.device)
                self.assertEqual(actual.requires_grad, expected.requires_grad)


if __name__ == "__main__":
    unittest.main()
