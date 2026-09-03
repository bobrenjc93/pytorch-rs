import subprocess
import sys
import unittest
from dataclasses import FrozenInstanceError, dataclass

import torch_rs as torch
from torch_rs import _compile_trace

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


REFERENCE_PYTORCH_VERSION = "2.13.0"
COMPILE_CORPUS_VERSION = "torch_compile_corpus_v1"

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

UNSUPPORTED_MESSAGE = (
    "torch.compile(): graph capture, graph execution, and eager fallback are "
    "not supported; only argument binding, disable=True pass-through, and "
    "backend resolution are implemented"
)


def cpu_float32_unary_abs_neg(x):
    return x.neg().abs()


def cpu_float32_unary_inputs(module):
    return (
        module.tensor(
            [[-3.25, -0.0, 1.5], [2.0, -4.5, 0.25]],
            dtype=module.float32,
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
        self.assertEqual(COMPILE_CORPUS_VERSION, "torch_compile_corpus_v1")
        self.assertEqual(sum(CATEGORY_WEIGHTS.values()), 100)
        self.assertEqual(len(COMPILE_CORPUS), 1)

        case = COMPILE_CORPUS[0]
        self.assertEqual(case.name, "cpu_float32_unary_abs_neg")
        self.assertEqual(case.category, "tensor_arithmetic")
        self.assertIn(case.category, CATEGORY_WEIGHTS)
        self.assertTrue(case.fullgraph)
        self.assertIsNone(case.dynamic)
        self.assertIsNone(case.mode)
        self.assertIsNone(case.options)
        self.assertNotIn("_compile_trace", torch.__all__)
        self.assertNotIn("_compile_trace_tensor_metadata", torch._C.__all__)
        self.assertNotIn("_compile_trace_grad_enabled", torch._C.__all__)
        self.assertNotIn("_compile_trace_unary", torch._C.__all__)


class CompileCorpusTraceTests(unittest.TestCase):
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

    def test_proxy_unsupported_operations_fail_clearly(self):
        recorder = _compile_trace.CompileTraceRecorder()
        x = recorder.input(shape=(2, 3))

        for operation, call in (
            ("Tensor.__add__", lambda: x + x),
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

def make_inputs(module):
    return (
        module.tensor(
            [[-3.25, -0.0, 1.5], [2.0, -4.5, 0.25]],
            dtype=module.float32,
        ),
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

compiled = torch.compile(program, backend=backend)
try:
    compiled(make_inputs(torch)[0])
except NotImplementedError as error:
    assert str(error) == (
        "torch.compile(): graph capture, graph execution, and eager fallback are "
        "not supported; only argument binding, disable=True pass-through, and "
        "backend resolution are implemented"
    )
else:
    raise AssertionError("public torch.compile should remain non-executing")
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

    def test_torch_rs_compile_keeps_eligible_cases_unsupported(self):
        for case in COMPILE_CORPUS:
            with self.subTest(case=case.name):
                self.assert_reference_eligible(case)

                model_calls = []
                backend_calls = []
                backend = make_recording_backend(backend_calls)

                def model(*args, **kwargs):
                    model_calls.append((args, kwargs))
                    return case.program(*args, **kwargs)

                compiled = torch.compile(model, **case.compile_kwargs(backend))
                self.assertIs(compiled._torch_rs_compile_backend, backend)
                self.assertEqual(model_calls, [])
                self.assertEqual(backend_calls, [])

                with self.assertRaises(NotImplementedError) as raised:
                    compiled(*case.make_inputs(torch))
                self.assertEqual(str(raised.exception), UNSUPPORTED_MESSAGE)
                self.assertEqual(model_calls, [])
                self.assertEqual(backend_calls, [])


if __name__ == "__main__":
    unittest.main()
