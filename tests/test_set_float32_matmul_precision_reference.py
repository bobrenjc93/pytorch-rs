import copy
import importlib
import inspect
import pickle
import pickletools
import types
import typing
import unittest
from unittest.mock import Mock

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class SetFloat32MatmulPrecisionReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "set_float32_matmul_precision differentials require pinned "
                "PyTorch 2.13.0"
            )

    def setUp(self):
        self.reference_original = reference_torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        reference_torch.set_float32_matmul_precision("highest")

    def tearDown(self):
        torch.set_float32_matmul_precision("highest")
        reference_torch.set_float32_matmul_precision(self.reference_original)

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

    def highest_outcome(self, module, precision, *, keyword=False):
        before = module.is_grad_enabled()
        if keyword:
            result = module.set_float32_matmul_precision(precision=precision)
        else:
            result = module.set_float32_matmul_precision(precision)
        left = module.tensor([[1.0, 2.0], [3.0, 4.0]])
        right = module.tensor([[5.0, 6.0], [7.0, 8.0]])
        product = module.matmul(left, right)
        return (
            result is None,
            module.get_float32_matmul_precision(),
            product.tolist(),
            str(product.dtype),
            str(product.device),
            before,
            module.is_grad_enabled(),
        )

    def test_highest_noop_and_bytes_compatibility_match_pytorch_2_13(self):
        class StringPrecision(str):
            def __eq__(self, other):
                raise AssertionError("string equality must not be dispatched")

            def __str__(self):
                raise AssertionError("string conversion must not be dispatched")

            def encode(self, *args, **kwargs):
                raise AssertionError("string encoding must not be dispatched")

        class BytesPrecision(bytes):
            def __eq__(self, other):
                raise AssertionError("bytes equality must not be dispatched")

            def decode(self, *args, **kwargs):
                raise AssertionError("bytes decoding must not be dispatched")

        for precision, keyword in (
            ("highest", False),
            ("highest", True),
            (b"highest", False),
            (StringPrecision("highest"), False),
            (BytesPrecision(b"highest"), False),
        ):
            with self.subTest(precision=precision, keyword=keyword):
                self.assertEqual(
                    self.highest_outcome(torch, precision, keyword=keyword),
                    self.highest_outcome(reference_torch, precision, keyword=keyword),
                )

        with torch.no_grad(), reference_torch.no_grad():
            self.assertEqual(
                self.highest_outcome(torch, "highest"),
                self.highest_outcome(reference_torch, "highest"),
            )

    def test_callable_metadata_matches_pytorch_2_13(self):
        actual_module = importlib.import_module("torch_rs")
        expected_module = importlib.import_module("torch")
        actual = actual_module.set_float32_matmul_precision
        expected = expected_module.set_float32_matmul_precision

        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual)), str(inspect.signature(expected))
        )
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(typing.get_type_hints(actual), typing.get_type_hints(expected))
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
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
        actual = torch.set_float32_matmul_precision
        expected = reference_torch.set_float32_matmul_precision

        self.assertEqual(
            torch.__all__.count("set_float32_matmul_precision"),
            reference_torch.__all__.count("set_float32_matmul_precision"),
        )
        for module, function in ((torch, actual), (reference_torch, expected)):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertIs(namespace["set_float32_matmul_precision"], function)
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

        self.assertFalse(hasattr(torch._C, "_set_float32_matmul_precision"))
        self.assertTrue(hasattr(reference_torch._C, "_set_float32_matmul_precision"))

    def test_argument_binding_errors_match_pytorch_2_13(self):
        actual = torch.set_float32_matmul_precision
        expected = reference_torch.set_float32_matmul_precision
        cases = (
            (lambda: actual(), lambda: expected()),
            (
                lambda: actual("highest", "highest"),
                lambda: expected("highest", "highest"),
            ),
            (
                lambda: actual(value="highest"),
                lambda: expected(value="highest"),
            ),
            (
                lambda: actual("highest", precision="highest"),
                lambda: expected("highest", precision="highest"),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)
                self.assertEqual(torch.get_float32_matmul_precision(), "highest")
                self.assertEqual(
                    reference_torch.get_float32_matmul_precision(), "highest"
                )

    def test_non_string_errors_match_pytorch_2_13(self):
        class CustomPrecision:
            def __str__(self):
                return "highest"

        shared_values = (
            None,
            True,
            1,
            1.0,
            object(),
            [],
            bytearray(b"highest"),
            memoryview(b"highest"),
            CustomPrecision(),
            "\ud800",
        )
        for value in shared_values:
            with self.subTest(value=ascii(value)):
                self.assert_error_matches(
                    lambda value=value: torch.set_float32_matmul_precision(value),
                    lambda value=value: reference_torch.set_float32_matmul_precision(
                        value
                    ),
                )
                self.assertEqual(torch.get_float32_matmul_precision(), "highest")
                self.assertEqual(
                    reference_torch.get_float32_matmul_precision(), "highest"
                )

        native_values = (
            (torch.tensor(1.0), reference_torch.tensor(1.0)),
            (torch.float32, reference_torch.float32),
            (torch.device("cpu"), reference_torch.device("cpu")),
        )
        for case, (actual_value, expected_value) in enumerate(native_values):
            with self.subTest(native_case=case):
                self.assert_error_matches(
                    lambda value=actual_value: torch.set_float32_matmul_precision(
                        value
                    ),
                    lambda value=expected_value: reference_torch.set_float32_matmul_precision(
                        value
                    ),
                )
                self.assertEqual(torch.get_float32_matmul_precision(), "highest")
                self.assertEqual(
                    reference_torch.get_float32_matmul_precision(), "highest"
                )

    def test_spoofed_or_raising_class_errors_match_pytorch_2_13(self):
        class_reads = []

        class SpoofedPrecision:
            @property
            def __class__(self):
                class_reads.append("spoofed")
                return str

        class RaisingPrecision:
            @property
            def __class__(self):
                class_reads.append("raising")
                raise AssertionError("the setter must not read __class__")

        values = (
            Mock(spec=str),
            Mock(spec=bytes),
            SpoofedPrecision(),
            RaisingPrecision(),
        )
        for case, value in enumerate(values):
            with self.subTest(case=case, type_name=type(value).__name__):
                self.assert_error_matches(
                    lambda value=value: torch.set_float32_matmul_precision(value),
                    lambda value=value: reference_torch.set_float32_matmul_precision(
                        value
                    ),
                )
                self.assertEqual(torch.get_float32_matmul_precision(), "highest")
                self.assertEqual(
                    reference_torch.get_float32_matmul_precision(), "highest"
                )

        self.assertEqual(class_reads, [])

    def test_reduced_precision_modes_remain_an_explicit_difference(self):
        for precision in ("high", "medium", b"high", b"medium"):
            with self.subTest(precision=precision):
                with self.assertRaises(NotImplementedError):
                    torch.set_float32_matmul_precision(precision)
                self.assertEqual(torch.get_float32_matmul_precision(), "highest")

                self.assertIsNone(
                    reference_torch.set_float32_matmul_precision(precision)
                )
                expected_precision = (
                    precision.decode() if isinstance(precision, bytes) else precision
                )
                self.assertEqual(
                    reference_torch.get_float32_matmul_precision(),
                    expected_precision,
                )
                reference_torch.set_float32_matmul_precision("highest")


if __name__ == "__main__":
    unittest.main()
