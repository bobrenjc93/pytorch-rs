import copy
import importlib
import inspect
import pickle
import pickletools
import threading
import types
import typing
import unittest
import warnings

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
                "float32 matmul precision differentials require pinned PyTorch "
                "2.13.0"
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

    def warning_outcome(self, module, precision):
        module.set_float32_matmul_precision("medium")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = module.set_float32_matmul_precision(precision)
            except BaseException as error:
                call = (
                    "error",
                    type(error).__module__,
                    type(error).__qualname__,
                    str(error),
                    error.args,
                )
            else:
                call = ("return", result is None)
        return (
            call,
            module.get_float32_matmul_precision(),
            tuple(
                (
                    item.category.__module__,
                    item.category.__qualname__,
                    type(item.message).__module__,
                    type(item.message).__qualname__,
                    str(item.message),
                )
                for item in caught
            ),
        )

    def threaded_outcome(self, module):
        module.set_float32_matmul_precision("high")
        observations = []
        errors = []

        def worker():
            try:
                with module.no_grad():
                    observations.append(
                        (module.is_grad_enabled(), module.get_float32_matmul_precision())
                    )
                    observations.append(
                        module.set_float32_matmul_precision("medium") is None
                    )
                    observations.append(
                        (module.is_grad_enabled(), module.get_float32_matmul_precision())
                    )
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        return (
            observations,
            errors,
            module.get_float32_matmul_precision(),
            module.is_grad_enabled(),
        )

    def matmul_outcome(self, module, precision):
        result = module.set_float32_matmul_precision(precision)
        left = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        right = module.tensor(
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], requires_grad=True
        )
        output = module.matmul(left, right)
        try:
            output.sum().backward()
        except BaseException as error:
            backward = (type(error).__name__, str(error))
        else:
            backward = None

        def grad_values(tensor):
            return None if tensor.grad is None else tensor.grad.tolist()

        return (
            result is None,
            module.get_float32_matmul_precision(),
            output.tolist(),
            tuple(output.shape),
            output.stride(),
            str(output.dtype),
            str(output.device),
            output.requires_grad,
            output.is_leaf,
            backward,
            grad_values(left),
            grad_values(right),
        )

    def test_supported_values_and_return_types_match_pytorch_2_13(self):
        for actual_precision, expected_precision in (
            ("highest", "highest"),
            ("high", "high"),
            ("medium", "medium"),
            (b"highest", b"highest"),
            (b"high", b"high"),
            (b"medium", b"medium"),
        ):
            with self.subTest(precision=actual_precision):
                actual_result = torch.set_float32_matmul_precision(actual_precision)
                expected_result = reference_torch.set_float32_matmul_precision(
                    expected_precision
                )
                self.assertIs(actual_result, expected_result)
                self.assertEqual(
                    torch.get_float32_matmul_precision(),
                    reference_torch.get_float32_matmul_precision(),
                )
                self.assertIs(type(torch.get_float32_matmul_precision()), str)

        self.assertIsNone(torch.set_float32_matmul_precision(precision="medium"))
        self.assertIsNone(
            reference_torch.set_float32_matmul_precision(precision="medium")
        )
        self.assertEqual(torch.get_float32_matmul_precision(), "medium")
        self.assertEqual(reference_torch.get_float32_matmul_precision(), "medium")

    def test_threaded_process_global_state_matches_pytorch_2_13(self):
        self.assertEqual(
            self.threaded_outcome(torch),
            self.threaded_outcome(reference_torch),
        )

    def test_callable_metadata_and_documentation_match_pytorch_2_13(self):
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

    def test_exports_copy_pickle_and_native_hooks_match_pytorch_2_13(self):
        names = (
            "get_float32_matmul_precision",
            "set_float32_matmul_precision",
        )
        for name in names:
            self.assertEqual(
                torch.__all__.count(name), reference_torch.__all__.count(name)
            )
            actual = getattr(torch, name)
            expected = getattr(reference_torch, name)
            for module, function in ((torch, actual), (reference_torch, expected)):
                namespace = {}
                exec(f"from {module.__name__} import *", namespace)
                self.assertIs(namespace[name], function)
                self.assertIs(copy.copy(function), function)
                self.assertIs(copy.deepcopy(function), function)

            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(name=name, protocol=protocol):
                    self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                    self.assertIs(
                        pickle.loads(pickle.dumps(expected, protocol)), expected
                    )
                    self.assertEqual(
                        self.pickle_shape(actual, protocol),
                        self.pickle_shape(expected, protocol),
                    )

        for module in (torch, reference_torch):
            self.assertTrue(hasattr(module._C, "_get_float32_matmul_precision"))
            self.assertTrue(hasattr(module._C, "_set_float32_matmul_precision"))

    def test_argument_binding_errors_and_state_preservation_match_pytorch_2_13(self):
        cases = (
            (
                lambda: torch.get_float32_matmul_precision(None),
                lambda: reference_torch.get_float32_matmul_precision(None),
            ),
            (
                lambda: torch.get_float32_matmul_precision(precision=None),
                lambda: reference_torch.get_float32_matmul_precision(precision=None),
            ),
            (
                lambda: torch.set_float32_matmul_precision(),
                lambda: reference_torch.set_float32_matmul_precision(),
            ),
            (
                lambda: torch.set_float32_matmul_precision("high", "medium"),
                lambda: reference_torch.set_float32_matmul_precision(
                    "high", "medium"
                ),
            ),
            (
                lambda: torch.set_float32_matmul_precision(value="high"),
                lambda: reference_torch.set_float32_matmul_precision(value="high"),
            ),
            (
                lambda: torch.set_float32_matmul_precision(
                    "high", precision="medium"
                ),
                lambda: reference_torch.set_float32_matmul_precision(
                    "high", precision="medium"
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                torch.set_float32_matmul_precision("medium")
                reference_torch.set_float32_matmul_precision("medium")
                self.assert_error_matches(actual_call, expected_call)
                self.assertEqual(torch.get_float32_matmul_precision(), "medium")
                self.assertEqual(
                    reference_torch.get_float32_matmul_precision(), "medium"
                )

    def test_invalid_types_and_utf8_errors_match_pytorch_2_13(self):
        class CustomPrecision:
            pass

        values = (
            None,
            True,
            1,
            1.0,
            object(),
            [],
            bytearray(b"high"),
            memoryview(b"high"),
            CustomPrecision(),
            "\ud800",
            b"\xff",
        )
        for case, value in enumerate(values):
            with self.subTest(case=case, value_type=type(value).__name__):
                self.assertEqual(
                    self.warning_outcome(torch, value),
                    self.warning_outcome(reference_torch, value),
                )

    def test_invalid_string_warnings_and_noops_match_pytorch_2_13(self):
        for precision in (
            "HIGH",
            "low",
            "",
            " medium ",
            "héllo",
            "😀",
            b"low",
            "highest\0ignored",
            "\0medium",
        ):
            with self.subTest(precision=precision):
                self.assertEqual(
                    self.warning_outcome(torch, precision),
                    self.warning_outcome(reference_torch, precision),
                )

    def test_cpu_matmul_values_and_each_implementations_gradients_ignore_modes(self):
        actual = [
            self.matmul_outcome(torch, precision)
            for precision in ("highest", "high", "medium")
        ]
        expected = [
            self.matmul_outcome(reference_torch, precision)
            for precision in ("highest", "high", "medium")
        ]

        for outcomes in (actual, expected):
            for precision, outcome in zip(
                ("highest", "high", "medium"), outcomes, strict=True
            ):
                self.assertTrue(outcome[0])
                self.assertEqual(outcome[1], precision)
                self.assertEqual(outcome[2:], outcomes[0][2:])
        self.assertEqual(actual[0][2:7], expected[0][2:7])
        self.assertEqual(actual[0][2], [[58.0, 64.0], [139.0, 154.0]])
        self.assertFalse(actual[0][7])
        self.assertTrue(actual[0][8])
        self.assertEqual(
            actual[0][9],
            (
                "RuntimeError",
                "element 0 of tensors does not require grad and does not have a grad_fn",
            ),
        )
        self.assertIsNone(actual[0][10])
        self.assertIsNone(actual[0][11])


if __name__ == "__main__":
    unittest.main()
