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


class CompileCorpusTraceTests(unittest.TestCase):
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
