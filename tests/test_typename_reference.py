import copy
import inspect
import pickle
import types
import typing
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class ExampleClass:
    class NestedClass:
        pass

    def method(self):
        return None


def example_function():
    return None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TypenameReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("typename differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def test_tensor_results_match_pytorch_2_13(self):
        actual_leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        expected_leaf = reference_torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        actual_tracked = (actual_leaf * 2.0).transpose(0, 1)
        expected_tracked = (expected_leaf * 2.0).transpose(0, 1)
        actual_tracked.sum().backward()
        expected_tracked.sum().backward()

        actual_tensors = (
            torch.tensor(1.0),
            torch.tensor([1.0, 2.0]),
            torch.zeros((2, 0, 3)),
            torch.zeros((2, 3, 4)).transpose(0, 2)[1],
            actual_leaf,
            actual_tracked,
            actual_leaf.grad,
        )
        expected_tensors = (
            reference_torch.tensor(1.0, dtype=reference_torch.float32),
            reference_torch.tensor([1.0, 2.0], dtype=reference_torch.float32),
            reference_torch.zeros((2, 0, 3)),
            reference_torch.zeros((2, 3, 4)).transpose(0, 2)[1],
            expected_leaf,
            expected_tracked,
            expected_leaf.grad,
        )

        for case, (actual, expected) in enumerate(
            zip(actual_tensors, expected_tensors, strict=True)
        ):
            with self.subTest(case=case):
                actual_result = torch.typename(actual)
                expected_result = reference_torch.typename(expected)
                self.assertIs(type(actual_result), type(expected_result))
                self.assertEqual(actual_result, expected_result)

    def test_generic_object_results_match_pytorch_2_13(self):
        instance = ExampleClass()
        objects = (
            None,
            True,
            1,
            1.5,
            2.0j,
            "value",
            b"value",
            [],
            (),
            {},
            object,
            object(),
            len,
            int,
            ExampleClass,
            ExampleClass.NestedClass,
            example_function,
            ExampleClass.method,
            instance.method,
            instance,
        )

        for case, value in enumerate(objects):
            with self.subTest(case=case, value=repr(value)):
                actual_result = torch.typename(value)
                expected_result = reference_torch.typename(value)
                self.assertIs(type(actual_result), type(expected_result))
                self.assertEqual(actual_result, expected_result)

    def test_package_owned_names_match_after_package_substitution(self):
        pairs = (
            (torch, reference_torch),
            (torch.typename, reference_torch.typename),
            (torch.Tensor, reference_torch.Tensor),
            (torch.float32, reference_torch.float32),
            (torch.device("cpu"), reference_torch.device("cpu")),
        )
        for actual, expected in pairs:
            with self.subTest(actual=repr(actual), expected=repr(expected)):
                self.assertEqual(
                    torch.typename(actual).replace("torch_rs", "torch"),
                    reference_torch.typename(expected),
                )

    def test_callable_metadata_matches_pytorch_2_13(self):
        actual = torch.typename
        expected = reference_torch.typename

        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(inspect.get_annotations(actual), inspect.get_annotations(expected))
        self.assertEqual(typing.get_type_hints(actual), typing.get_type_hints(expected))
        self.assertEqual(inspect.signature(actual), inspect.signature(expected))
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )

    def test_exports_copy_and_pickle_match_pytorch_2_13(self):
        actual = torch.typename
        expected = reference_torch.typename

        self.assertEqual(
            torch.__all__.count("typename"),
            reference_torch.__all__.count("typename"),
        )
        for module, function in ((torch, actual), (reference_torch, expected)):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertIs(namespace["typename"], function)
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(pickle.loads(pickle.dumps(expected, protocol)), expected)

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        cases = (
            (lambda: torch.typename(), lambda: reference_torch.typename()),
            (
                lambda: torch.typename(actual, actual),
                lambda: reference_torch.typename(expected, expected),
            ),
            (
                lambda: torch.typename(obj=actual),
                lambda: reference_torch.typename(obj=expected),
            ),
            (
                lambda: torch.typename(actual, obj=actual),
                lambda: reference_torch.typename(expected, obj=expected),
            ),
            (
                lambda: torch.typename(input=actual),
                lambda: reference_torch.typename(input=expected),
            ),
            (
                lambda: torch.typename(extra=actual),
                lambda: reference_torch.typename(extra=expected),
            ),
            (
                lambda: torch.typename(actual, extra=actual),
                lambda: reference_torch.typename(expected, extra=expected),
            ),
            (
                lambda: torch.typename(obj=actual, extra=actual),
                lambda: reference_torch.typename(obj=expected, extra=expected),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_legacy_tensor_constructors_and_type_conversion_stay_out_of_scope(self):
        legacy_types = (
            "ByteTensor",
            "CharTensor",
            "ShortTensor",
            "IntTensor",
            "LongTensor",
            "HalfTensor",
            "FloatTensor",
            "DoubleTensor",
            "BoolTensor",
            "BFloat16Tensor",
        )
        for name in legacy_types:
            with self.subTest(name=name):
                self.assertTrue(hasattr(reference_torch, name))
                self.assertFalse(hasattr(torch, name))
                self.assertNotIn(name, torch.__all__)
        self.assertTrue(hasattr(reference_torch.Tensor, "type"))
        self.assertFalse(hasattr(torch.Tensor, "type"))


if __name__ == "__main__":
    unittest.main()
