import contextlib
import copy
import importlib
import inspect
import pickle
import pickletools
import threading
import types
import typing
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class GetFloat32MatmulPrecisionReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "get_float32_matmul_precision differentials require pinned "
                "PyTorch 2.13.0"
            )

    def setUp(self):
        self.actual_original = torch.backends.cuda.matmul.allow_tf32
        self.expected_original = reference_torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        reference_torch.set_float32_matmul_precision("highest")

    def tearDown(self):
        torch.backends.cuda.matmul.allow_tf32 = self.actual_original
        reference_torch.set_float32_matmul_precision(self.expected_original)

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def supported_state_outcome(self, module):
        function = module.get_float32_matmul_precision

        def query_outcome():
            before = module.is_grad_enabled()
            first = function()
            middle = module.is_grad_enabled()
            second = function()
            after = module.is_grad_enabled()
            return (
                before,
                type(first) is str,
                first,
                middle,
                type(second) is str,
                second,
                after,
            )

        states = [query_outcome()]
        with module.no_grad():
            states.append(query_outcome())
            with module.no_grad():
                states.append(query_outcome())
            states.append(query_outcome())
        states.append(query_outcome())

        worker_count = 8
        barrier = threading.Barrier(worker_count)
        worker_states = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = module.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    worker_states[index] = query_outcome()
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))

        threads = [
            threading.Thread(target=worker, args=(index,))
            for index in range(worker_count)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        return states, worker_states

    def pickle_shape(self, function, protocol):
        shape = []
        for opcode, argument, _ in pickletools.genops(
            pickle.dumps(function, protocol=protocol)
        ):
            if opcode.name == "FRAME":
                argument = "<frame length>"
            elif isinstance(argument, str):
                argument = argument.replace("torch_rs", "torch")
            shape.append((opcode.name, argument))
        return shape

    def test_supported_shared_threaded_and_grad_states_match_pytorch_2_13(self):
        for allow_tf32 in (False, True):
            with self.subTest(allow_tf32=allow_tf32):
                torch.backends.cuda.matmul.allow_tf32 = allow_tf32
                reference_torch.backends.cuda.matmul.allow_tf32 = allow_tf32
                self.assertEqual(
                    self.supported_state_outcome(torch),
                    self.supported_state_outcome(reference_torch),
                )

        torch.set_float32_matmul_precision("highest")
        reference_torch.set_float32_matmul_precision("highest")
        self.assertEqual(
            self.supported_state_outcome(torch),
            self.supported_state_outcome(reference_torch),
        )

    def test_reference_only_setter_bounds_unsupported_reduced_precision_states(self):
        actual = torch.get_float32_matmul_precision
        expected = reference_torch.get_float32_matmul_precision
        actual_states = []
        expected_states = []
        for precision in ("highest", "high", "medium", "highest"):
            reference_torch.set_float32_matmul_precision(precision)
            actual_states.append(actual())
            expected_states.append(expected())

        self.assertEqual(actual_states, ["highest"] * 4)
        self.assertEqual(expected_states, ["highest", "high", "medium", "highest"])
        for state in (*actual_states, *expected_states):
            self.assertIs(type(state), str)

    def test_signature_annotations_documentation_and_identity_match(self):
        actual_module = importlib.import_module("torch_rs")
        expected_module = importlib.import_module("torch")
        actual = actual_module.get_float32_matmul_precision
        expected = expected_module.get_float32_matmul_precision

        self.assertIs(torch, actual_module)
        self.assertIs(reference_torch, expected_module)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(str(inspect.signature(actual)), str(inspect.signature(expected)))
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(typing.get_type_hints(actual), typing.get_type_hints(expected))
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"),
            expected.__module__,
        )
        self.assertIs(inspect.getmodule(actual), actual_module)
        self.assertIs(inspect.getmodule(expected), expected_module)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )

    def test_exports_copy_and_pickle_match_pytorch_2_13(self):
        actual = torch.get_float32_matmul_precision
        expected = reference_torch.get_float32_matmul_precision

        self.assertEqual(
            torch.__all__.count("get_float32_matmul_precision"),
            reference_torch.__all__.count("get_float32_matmul_precision"),
        )
        for module, function in ((torch, actual), (reference_torch, expected)):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertIs(namespace["get_float32_matmul_precision"], function)
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(pickle.loads(pickle.dumps(expected, protocol)), expected)
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.get_float32_matmul_precision
        expected = reference_torch.get_float32_matmul_precision
        cases = (
            (lambda: actual(None), lambda: expected(None)),
            (lambda: actual(None, None), lambda: expected(None, None)),
            (
                lambda: actual(precision=None),
                lambda: expected(precision=None),
            ),
            (
                lambda: actual(None, precision=None),
                lambda: expected(None, precision=None),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_highest_only_setter_is_an_explicit_python_layer_subset(self):
        self.assertTrue(hasattr(reference_torch, "set_float32_matmul_precision"))
        self.assertIn("set_float32_matmul_precision", reference_torch.__all__)
        self.assertTrue(hasattr(torch, "set_float32_matmul_precision"))
        self.assertEqual(torch.__all__.count("set_float32_matmul_precision"), 1)
        self.assertFalse(hasattr(torch._C, "_set_float32_matmul_precision"))
        self.assertTrue(hasattr(reference_torch._C, "_set_float32_matmul_precision"))


if __name__ == "__main__":
    unittest.main()
