import copy
import importlib
import inspect
import os
import pickle
import pickletools
import threading
import types
import typing
import unittest
import warnings

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class Float32MatmulPrecisionReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "float32 matmul precision differentials require pinned "
                "PyTorch 2.13.0"
            )

    def setUp(self):
        self.original_precisions = {
            module: module.get_float32_matmul_precision()
            for module in (torch, reference_torch)
        }
        for module in self.original_precisions:
            module.set_float32_matmul_precision("highest")

    def tearDown(self):
        for module, precision in self.original_precisions.items():
            module.set_float32_matmul_precision(precision)

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def mutation_outcome(self, module):
        outcomes = []
        for precision in (
            "highest",
            "high",
            "medium",
            b"highest",
            b"high",
            b"medium",
        ):
            result = module.set_float32_matmul_precision(precision)
            state = module.get_float32_matmul_precision()
            outcomes.append((result is None, type(state), state))
        return outcomes

    def threaded_outcome(self, module):
        worker_ready = threading.Event()
        read_updated = threading.Event()
        observations = []
        errors = []
        module.set_float32_matmul_precision("high")

        def observer():
            try:
                observations.append(module.get_float32_matmul_precision())
                worker_ready.set()
                if not read_updated.wait(timeout=10):
                    raise RuntimeError("timed out waiting for precision update")
                observations.append(module.get_float32_matmul_precision())
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))

        thread = threading.Thread(target=observer)
        thread.start()
        self.assertTrue(worker_ready.wait(timeout=10))
        result = module.set_float32_matmul_precision("medium")
        read_updated.set()
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        return result is None, observations, errors, module.get_float32_matmul_precision()

    def warning_outcome(self, module, value):
        module.set_float32_matmul_precision("high")
        setter = module.set_float32_matmul_precision
        source_line = inspect.getsourcelines(setter)[1]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = setter(value)
        return (
            result is None,
            module.get_float32_matmul_precision(),
            tuple(
                (
                    warning.category,
                    str(warning.message),
                    os.path.basename(warning.filename),
                    warning.lineno - source_line,
                )
                for warning in caught
            ),
        )

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

    def test_supported_modes_and_thread_visibility_match_pytorch_2_13(self):
        self.assertEqual(self.mutation_outcome(torch), self.mutation_outcome(reference_torch))
        self.assertEqual(self.threaded_outcome(torch), self.threaded_outcome(reference_torch))

    def test_invalid_string_warnings_and_no_op_behavior_match_pytorch_2_13(self):
        for value in (
            "Highest",
            "low",
            "",
            "medium\x00ignored",
            b"highest\x00ignored",
            "é",
        ):
            with self.subTest(value=repr(value)):
                self.assertEqual(
                    self.warning_outcome(torch, value),
                    self.warning_outcome(reference_torch, value),
                )

        for module in (torch, reference_torch):
            module.set_float32_matmul_precision("medium")
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                with self.assertRaises(UserWarning):
                    module.set_float32_matmul_precision("unsupported")
            self.assertEqual(module.get_float32_matmul_precision(), "medium")

    def test_value_validation_errors_match_pytorch_2_13(self):
        class StringConvertible:
            def __str__(self):
                return "high"

        common_values = (
            None,
            True,
            1,
            1.5,
            bytearray(b"high"),
            memoryview(b"high"),
            [],
            {},
            object(),
            StringConvertible(),
            "\ud800",
            b"\xff",
        )
        value_pairs = [(value, value) for value in common_values] + [
            (torch.float32, reference_torch.float32),
            (torch.device("cpu"), reference_torch.device("cpu")),
            (
                np.array([1.0], dtype=np.float32),
                np.array([1.0], dtype=np.float32),
            ),
        ]
        for actual_value, expected_value in value_pairs:
            with self.subTest(
                actual_type=type(actual_value).__name__,
                expected_type=type(expected_value).__name__,
            ):
                torch.set_float32_matmul_precision("medium")
                reference_torch.set_float32_matmul_precision("medium")
                self.assert_error_matches(
                    lambda value=actual_value: torch.set_float32_matmul_precision(
                        value
                    ),
                    lambda value=expected_value: (
                        reference_torch.set_float32_matmul_precision(value)
                    ),
                )
                self.assertEqual(torch.get_float32_matmul_precision(), "medium")
                self.assertEqual(
                    reference_torch.get_float32_matmul_precision(), "medium"
                )

    def test_signature_annotations_documentation_and_identity_match(self):
        actual_module = importlib.import_module("torch_rs")
        expected_module = importlib.import_module("torch")
        for name in (
            "get_float32_matmul_precision",
            "set_float32_matmul_precision",
        ):
            with self.subTest(name=name):
                actual = getattr(actual_module, name)
                expected = getattr(expected_module, name)
                self.assertIs(type(actual), types.FunctionType)
                self.assertIs(type(expected), types.FunctionType)
                self.assertEqual(
                    str(inspect.signature(actual)), str(inspect.signature(expected))
                )
                self.assertEqual(actual.__annotations__, expected.__annotations__)
                self.assertEqual(
                    typing.get_type_hints(actual), typing.get_type_hints(expected)
                )
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
        for name in (
            "get_float32_matmul_precision",
            "set_float32_matmul_precision",
        ):
            with self.subTest(name=name):
                actual = getattr(torch, name)
                expected = getattr(reference_torch, name)
                self.assertEqual(
                    torch.__all__.count(name), reference_torch.__all__.count(name)
                )
                for module, function in (
                    (torch, actual),
                    (reference_torch, expected),
                ):
                    namespace = {}
                    exec(f"from {module.__name__} import *", namespace)
                    self.assertIs(namespace[name], function)
                    self.assertIs(copy.copy(function), function)
                    self.assertIs(copy.deepcopy(function), function)

                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    self.assertIs(
                        pickle.loads(pickle.dumps(actual, protocol)), actual
                    )
                    self.assertIs(
                        pickle.loads(pickle.dumps(expected, protocol)), expected
                    )
                    self.assertEqual(
                        self.pickle_shape(actual, protocol),
                        self.pickle_shape(expected, protocol),
                    )

        for name in (
            "_get_float32_matmul_precision",
            "_set_float32_matmul_precision",
        ):
            self.assertTrue(hasattr(torch._C, name))
            self.assertTrue(hasattr(reference_torch._C, name))
            self.assertNotIn(name, torch._C.__all__)

    def test_argument_errors_and_state_preservation_match_pytorch_2_13(self):
        actual_getter = torch.get_float32_matmul_precision
        expected_getter = reference_torch.get_float32_matmul_precision
        actual_setter = torch.set_float32_matmul_precision
        expected_setter = reference_torch.set_float32_matmul_precision
        cases = (
            (lambda: actual_getter(None), lambda: expected_getter(None)),
            (
                lambda: actual_getter(precision=None),
                lambda: expected_getter(precision=None),
            ),
            (lambda: actual_setter(), lambda: expected_setter()),
            (
                lambda: actual_setter("high", "medium"),
                lambda: expected_setter("high", "medium"),
            ),
            (
                lambda: actual_setter(mode="high"),
                lambda: expected_setter(mode="high"),
            ),
            (
                lambda: actual_setter("high", precision="medium"),
                lambda: expected_setter("high", precision="medium"),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                actual_before = actual_getter()
                expected_before = expected_getter()
                self.assert_error_matches(actual_call, expected_call)
                self.assertEqual(actual_getter(), actual_before)
                self.assertEqual(expected_getter(), expected_before)

        self.assertIsNone(actual_setter(precision="high"))
        self.assertIsNone(expected_setter(precision="high"))
        self.assertEqual(actual_getter(), expected_getter())


if __name__ == "__main__":
    unittest.main()
