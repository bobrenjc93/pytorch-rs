from dataclasses import dataclass
import unittest

import torch_rs as torch

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
