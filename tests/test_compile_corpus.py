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
    return x.neg().abs() + y.negative()


def cpu_float32_matrix_vector_add_method(x, y):
    return x.add(y.abs())


def cpu_float32_tensor_scalar_add(x, y):
    return (x + y).abs()


def cpu_float32_scalar_tensor_add(x, y):
    return x.add(y.neg())


def cpu_float32_heldout_broadcast_chain(x, y):
    z = y.abs()
    return (x + z).neg().add(y)


def cpu_float32_heldout_scalar_left_broadcast(x, y):
    return (x.neg() + y).absolute()


def cpu_float32_matrix_vector_inputs(module):
    return (
        module.tensor(
            [[-3.0, 0.5, 4.0], [2.25, -5.5, 6.75]],
            dtype=module.float32,
        ),
        module.tensor([1.0, -2.0, 0.25], dtype=module.float32),
    )


def cpu_float32_matrix_vector_requires_grad_inputs(module):
    return (
        module.tensor(
            [[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]],
            dtype=module.float32,
            requires_grad=True,
        ),
        module.tensor([0.5, -1.5, 2.5], dtype=module.float32),
    )


def cpu_float32_tensor_scalar_inputs(module):
    return (
        module.tensor(
            [[-2.0, 0.0, 3.5], [4.25, -5.75, 6.0]],
            dtype=module.float32,
        ),
        module.tensor(-1.25, dtype=module.float32, requires_grad=True),
    )


def cpu_float32_scalar_tensor_inputs(module):
    return (
        module.tensor(2.0, dtype=module.float32),
        module.tensor(
            [[-0.5, 1.5, -2.5], [3.5, -4.5, 5.5]],
            dtype=module.float32,
            requires_grad=True,
        ),
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
        name="cpu_float32_matrix_vector_add_method",
        category="broadcasting",
        program=cpu_float32_matrix_vector_add_method,
        make_inputs=cpu_float32_matrix_vector_requires_grad_inputs,
    ),
    CompileCorpusCase(
        name="cpu_float32_tensor_scalar_add",
        category="broadcasting",
        program=cpu_float32_tensor_scalar_add,
        make_inputs=cpu_float32_tensor_scalar_inputs,
    ),
    CompileCorpusCase(
        name="cpu_float32_scalar_tensor_add",
        category="broadcasting",
        program=cpu_float32_scalar_tensor_add,
        make_inputs=cpu_float32_scalar_tensor_inputs,
    ),
)


COMPILE_HELD_OUT_CORPUS = (
    CompileCorpusCase(
        name="cpu_float32_heldout_broadcast_chain",
        category="broadcasting",
        program=cpu_float32_heldout_broadcast_chain,
        make_inputs=cpu_float32_matrix_vector_inputs,
    ),
    CompileCorpusCase(
        name="cpu_float32_heldout_scalar_left_broadcast",
        category="broadcasting",
        program=cpu_float32_heldout_scalar_left_broadcast,
        make_inputs=cpu_float32_scalar_tensor_inputs,
    ),
)


def compile_corpus_cases(include_held_out=False):
    if include_held_out:
        return (*COMPILE_CORPUS, *COMPILE_HELD_OUT_CORPUS)
    return COMPILE_CORPUS


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
        self.assertEqual(len(COMPILE_CORPUS), 9)
        self.assertEqual(len(COMPILE_HELD_OUT_CORPUS), 2)

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
                "cpu_float32_matrix_vector_add_method",
                "cpu_float32_tensor_scalar_add",
                "cpu_float32_scalar_tensor_add",
            ],
        )
        held_out_case_names = [case.name for case in COMPILE_HELD_OUT_CORPUS]
        self.assertEqual(
            held_out_case_names,
            [
                "cpu_float32_heldout_broadcast_chain",
                "cpu_float32_heldout_scalar_left_broadcast",
            ],
        )

        categories = {case.category for case in COMPILE_CORPUS}
        self.assertEqual(categories, {"tensor_arithmetic", "broadcasting"})
        for case in COMPILE_CORPUS:
            with self.subTest(case=case.name):
                self.assertIn(case.category, CATEGORY_WEIGHTS)
                self.assertTrue(case.fullgraph)
                self.assertIsNone(case.dynamic)
                self.assertIsNone(case.mode)
                self.assertIsNone(case.options)
        for case in COMPILE_HELD_OUT_CORPUS:
            with self.subTest(held_out_case=case.name):
                self.assertEqual(case.category, "broadcasting")
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
        self.assertFalse(hasattr(_compile_trace, "lower_compile_graph"))


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
            (
                cpu_float32_matrix_vector_add,
                cpu_float32_matrix_vector_inputs,
                ["neg", "abs", "neg", "add"],
            ),
            (
                cpu_float32_matrix_vector_add_method,
                cpu_float32_matrix_vector_requires_grad_inputs,
                ["abs", "add"],
            ),
            (
                cpu_float32_tensor_scalar_add,
                cpu_float32_tensor_scalar_inputs,
                ["add", "abs"],
            ),
            (
                cpu_float32_scalar_tensor_add,
                cpu_float32_scalar_tensor_inputs,
                ["neg", "add"],
            ),
        )
        for program, make_inputs, expected_targets in cases:
            with self.subTest(program=program.__name__):
                inputs = make_inputs(torch)
                graph = _compile_bytecode.lower_compile_graph(
                    program,
                    tuple(
                        _compile_trace._metadata_from_native_tensor(input)
                        for input in inputs
                    ),
                    name=program.__name__,
                )
                expected = program(*inputs)

                self.assertEqual(graph.name, program.__name__)
                self.assertEqual(
                    [input.name for input in graph.inputs],
                    list(program.__code__.co_varnames[: program.__code__.co_argcount]),
                )
                self.assertEqual(
                    [operation.target for operation in graph.operations],
                    expected_targets,
                )
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

    def test_bytecode_lowerer_rejects_unsupported_input_counts(self):
        def no_inputs():
            raise AssertionError("unsupported program should not run")

        def three_inputs(x, y, z):
            raise AssertionError("unsupported program should not run")

        input = torch.tensor([1.0], dtype=torch.float32)
        metadata = _compile_trace._metadata_from_native_tensor(input)
        for program, input_metadatas in (
            (no_inputs, ()),
            (three_inputs, (metadata, metadata, metadata)),
        ):
            with self.subTest(program=program.__name__):
                with self.assertRaisesRegex(
                    _compile_trace.CompileTraceUnsupportedError,
                    "one or two positional Tensor arguments",
                ):
                    _compile_bytecode.lower_compile_graph(
                        program,
                        input_metadatas,
                        name=program.__name__,
                    )

    def test_private_recorder_rejects_more_than_two_inputs(self):
        recorder = _compile_trace.CompileTraceRecorder()
        first = recorder.input(shape=(1,))
        second = recorder.input(shape=(1,))
        third = recorder.input(shape=(1,))

        with self.assertRaisesRegex(
            _compile_trace.CompileTraceUnsupportedError,
            "one or two inputs",
        ):
            recorder.finish((first + second) + third)

    def test_bytecode_lowerer_rejects_python_scalar_add_operands(self):
        def scalar_operand(x, y):
            return x + 1.0

        left, right = cpu_float32_matrix_vector_inputs(torch)
        with self.assertRaisesRegex(
            _compile_trace.CompileTraceUnsupportedError,
            "non-Tensor right operand",
        ):
            _compile_bytecode.lower_compile_graph(
                scalar_operand,
                (
                    _compile_trace._metadata_from_native_tensor(left),
                    _compile_trace._metadata_from_native_tensor(right),
                ),
                name="scalar_operand",
            )

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

    def test_bytecode_lowerer_accepts_combined_distinct_local_loads(self):
        def program(x, y):
            raise AssertionError("synthetic bytecode test must not run")

        left = torch.tensor(
            [[-2.0, 0.5, 3.0], [4.0, -5.0, 6.0]],
            dtype=torch.float32,
        )
        right = torch.tensor([1.0, -2.0, 0.25], dtype=torch.float32)
        instructions = (
            self.bytecode_instruction("LOAD_FAST_LOAD_FAST", ("x", "y"), ""),
            self.bytecode_instruction("BINARY_OP", argrepr="+"),
            self.bytecode_instruction("RETURN_VALUE"),
        )

        graph = self.lower_with_bytecode_instructions(
            program,
            instructions,
            left,
            right,
        )

        self.assertEqual(
            [operation.target for operation in graph.operations],
            ["add"],
        )
        self.assertEqual([input.name for input in graph.inputs], ["x", "y"])
        self.assertEqual(graph.operations[0].inputs, ("x", "y"))
        self.assert_native_tensor_matches(
            graph.forward(left, right),
            left + right,
            case="LOAD_FAST_LOAD_FAST distinct locals",
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

    def test_two_input_broadcast_records_private_immutable_graph(self):
        graph = _compile_trace.trace_compile_graph(
            cpu_float32_matrix_vector_add,
            cpu_float32_matrix_vector_inputs,
            name="cpu_float32_matrix_vector_add",
        )
        inputs = cpu_float32_matrix_vector_inputs(torch)
        expected = cpu_float32_matrix_vector_add(*inputs)

        self.assertEqual(graph.name, "cpu_float32_matrix_vector_add")
        self.assertEqual(graph.output, "add_3")
        self.assertEqual([input.name for input in graph.inputs], ["arg0", "arg1"])
        self.assertEqual([input.index for input in graph.inputs], [0, 1])
        self.assertEqual(len(graph.operations), 4)

        self.assertEqual(graph.inputs[0].metadata.shape, (2, 3))
        self.assertEqual(graph.inputs[0].metadata.stride, (3, 1))
        self.assertEqual(graph.inputs[1].metadata.shape, (3,))
        self.assertEqual(graph.inputs[1].metadata.stride, (1,))

        neg, abs_op, right_neg, add_op = graph.operations
        self.assertEqual(neg.target, "neg")
        self.assertEqual(neg.inputs, ("arg0",))
        self.assertEqual(abs_op.target, "abs")
        self.assertEqual(abs_op.inputs, ("neg_0",))
        self.assertEqual(right_neg.target, "neg")
        self.assertEqual(right_neg.inputs, ("arg1",))
        self.assertEqual(add_op.target, "add")
        self.assertEqual(add_op.inputs, ("abs_1", "neg_2"))
        self.assertEqual(add_op.metadata.shape, tuple(expected.shape))
        self.assertEqual(add_op.metadata.stride, expected.stride())
        self.assertEqual(graph.output_metadata, add_op.metadata)
        self.assert_native_tensor_matches(
            graph.forward(*inputs),
            expected,
            case="two-input graph",
        )

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

    def test_two_input_broadcast_executes_private_native_graph_for_key_layouts(self):
        cases = (
            (
                "matrix vector",
                cpu_float32_matrix_vector_add,
                cpu_float32_matrix_vector_inputs,
            ),
            (
                "method matrix vector requires_grad",
                cpu_float32_matrix_vector_add_method,
                cpu_float32_matrix_vector_requires_grad_inputs,
            ),
            (
                "tensor scalar",
                cpu_float32_tensor_scalar_add,
                cpu_float32_tensor_scalar_inputs,
            ),
            (
                "scalar tensor",
                cpu_float32_scalar_tensor_add,
                cpu_float32_scalar_tensor_inputs,
            ),
        )
        for case, program, make_inputs in cases:
            with self.subTest(case=case):
                graph = _compile_trace.trace_compile_graph(
                    program,
                    make_inputs,
                    name=program.__name__,
                )
                inputs = make_inputs(torch)
                expected = program(*inputs)

                self.assertEqual(len(graph.inputs), 2)
                self.assertEqual(graph.output_metadata.shape, tuple(expected.shape))
                self.assertEqual(graph.output_metadata.stride, expected.stride())
                self.assertEqual(
                    graph.output_metadata.requires_grad,
                    expected.requires_grad,
                )
                self.assert_native_tensor_matches(
                    graph.forward(*inputs),
                    expected,
                    case=case,
                )
                self.assert_native_tensor_matches(
                    _compile_trace.execute_compile_trace_graph(graph, *inputs),
                    expected,
                    case=f"{case} function executor",
                )

    def test_two_input_binary_executor_matches_empty_and_strided_broadcasts(self):
        cases = (
            (
                "strided matrix vector",
                torch.tensor(
                    [[[-3.0, 0.5], [4.0, -2.0], [1.25, 6.0]]],
                    dtype=torch.float32,
                ).transpose(0, 2),
                torch.tensor([[1.0], [-0.5], [2.0]], dtype=torch.float32),
            ),
            (
                "empty strided",
                torch.zeros((2, 0, 3), dtype=torch.float32).transpose(0, 2),
                torch.ones((1, 1, 2), dtype=torch.float32),
            ),
        )
        for case, left, right in cases:
            with self.subTest(case=case):
                recorder = _compile_trace.CompileTraceRecorder(name=case)
                left_proxy = recorder.input(
                    shape=tuple(left.shape),
                    stride=left.stride(),
                    dtype=_compile_trace.float32,
                    device="cpu",
                    requires_grad=left.requires_grad,
                )
                right_proxy = recorder.input(
                    shape=tuple(right.shape),
                    stride=right.stride(),
                    dtype=_compile_trace.float32,
                    device="cpu",
                    requires_grad=right.requires_grad,
                )
                graph = recorder.finish(left_proxy + right_proxy)
                expected = left + right

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
                    graph.forward(left, right),
                    expected,
                    case=case,
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

    def test_two_input_binary_executor_matches_no_grad_requires_grad_outputs(self):
        graph = _compile_trace.trace_compile_graph(
            cpu_float32_matrix_vector_add_method,
            cpu_float32_matrix_vector_requires_grad_inputs,
            name="cpu_float32_matrix_vector_add_method_requires_grad",
        )
        inputs = cpu_float32_matrix_vector_requires_grad_inputs(torch)
        expected_with_grad = cpu_float32_matrix_vector_add_method(*inputs)

        self.assertTrue(graph.inputs[0].metadata.requires_grad)
        self.assertFalse(graph.inputs[1].metadata.requires_grad)
        self.assertTrue(graph.operations[-1].metadata.requires_grad)
        self.assertTrue(graph.output_metadata.requires_grad)
        self.assert_native_tensor_matches(
            graph.forward(*inputs),
            expected_with_grad,
            case="two-input requires_grad grad-enabled",
        )

        with torch.no_grad():
            expected_no_grad = cpu_float32_matrix_vector_add_method(*inputs)
            actual_no_grad = graph.forward(*inputs)

        self.assertFalse(expected_no_grad.requires_grad)
        self.assert_native_tensor_matches(
            actual_no_grad,
            expected_no_grad,
            case="two-input requires_grad no_grad",
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

    def test_private_executor_rejects_two_input_runtime_metadata_mismatch(self):
        graph = _compile_trace.trace_compile_graph(
            cpu_float32_matrix_vector_add,
            cpu_float32_matrix_vector_inputs,
            name="cpu_float32_matrix_vector_add",
        )
        left, _ = cpu_float32_matrix_vector_inputs(torch)
        mismatched = torch.tensor([[1.0, -2.0, 3.0]], dtype=torch.float32)

        with self.assertRaisesRegex(
            ValueError,
            (
                "metadata mismatch for 'arg1': "
                r"shape expected \(3,\), got \(1, 3\)"
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

def broadcast_add(x, y):
    return x.neg().abs() + y.negative()

def make_inputs(module):
    return (
        module.tensor(
            [[-3.25, -0.0, 1.5], [2.0, -4.5, 0.25]],
            dtype=module.float32,
        ),
    )

def make_two_inputs(module):
    return (
        module.tensor(
            [[-3.0, 0.5, 4.0], [2.25, -5.5, 6.75]],
            dtype=module.float32,
        ),
        module.tensor([1.0, -2.0, 0.25], dtype=module.float32),
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

two_input_graph = _compile_trace.trace_compile_graph(
    broadcast_add,
    make_two_inputs,
    name="cpu_float32_matrix_vector_add",
)
assert two_input_graph.output == "add_3"
assert [operation.target for operation in two_input_graph.operations] == [
    "neg",
    "abs",
    "neg",
    "add",
]
assert two_input_graph.operations[-1].inputs == ("abs_1", "neg_2")
two_inputs = make_two_inputs(torch)
two_expected = broadcast_add(*two_inputs)
for actual in (
    two_input_graph.forward(*two_inputs),
    _compile_trace.execute_compile_trace_graph(two_input_graph, *two_inputs),
):
    assert actual.tolist() == two_expected.tolist()
    assert actual.shape == two_expected.shape
    assert actual.stride() == two_expected.stride()
    assert actual.dtype is two_expected.dtype
    assert actual.device == two_expected.device
    assert actual.requires_grad is two_expected.requires_grad
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
        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        self.assertEqual(actual.stride(), expected.stride())
        self.assertIs(actual.dtype, expected.dtype)
        self.assertEqual(actual.device, expected.device)
        self.assertEqual(actual.requires_grad, expected.requires_grad)
        self.assertGreaterEqual(len(backend_calls), 1)

    def test_reference_pytorch_2_13_accepts_all_eligible_cases(self):
        for case in compile_corpus_cases(include_held_out=True):
            with self.subTest(case=case.name):
                self.assert_reference_eligible(case)

    def test_torch_rs_compile_runs_eligible_eager_cases_natively(self):
        for case in compile_corpus_cases(include_held_out=True):
            with self.subTest(case=case.name):
                self.assert_reference_eligible(case)

                inputs = case.make_inputs(torch)
                expected = case.program(*case.make_inputs(torch))
                reference_expected = case.program(*case.make_inputs(reference_torch))
                compiled = torch.compile(
                    case.program,
                    **case.compile_kwargs("eager"),
                )
                actual = compiled(*inputs)
                self.assertIs(compiled._torch_rs_compile_backend, "eager")
                self.assertEqual(actual.tolist(), expected.tolist())
                self.assertEqual(actual.tolist(), reference_expected.tolist())
                self.assertEqual(tuple(actual.shape), tuple(expected.shape))
                self.assertEqual(tuple(actual.shape), tuple(reference_expected.shape))
                self.assertEqual(actual.stride(), expected.stride())
                self.assertEqual(actual.stride(), reference_expected.stride())
                self.assertIs(actual.dtype, expected.dtype)
                self.assertEqual(str(actual.dtype), str(reference_expected.dtype))
                self.assertEqual(actual.device, expected.device)
                self.assertEqual(str(actual.device), str(reference_expected.device))
                self.assertEqual(actual.requires_grad, expected.requires_grad)
                self.assertEqual(
                    actual.requires_grad,
                    reference_expected.requires_grad,
                )


if __name__ == "__main__":
    unittest.main()
