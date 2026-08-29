import copy
import importlib
import inspect
import pickle
import re
import types
import typing
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class ConversionTrap:
    def __init__(self):
        self.calls = []

    def called(self, name):
        self.calls.append(name)
        raise AssertionError(f"is_storage invoked {name}")

    def __torch_function__(self, *args, **kwargs):
        self.called("__torch_function__")

    def __torch_dispatch__(self, *args, **kwargs):
        self.called("__torch_dispatch__")

    def __array__(self, *args, **kwargs):
        self.called("__array__")

    @property
    def __array_interface__(self):
        self.called("__array_interface__")

    @property
    def __array_struct__(self):
        self.called("__array_struct__")

    @property
    def __cuda_array_interface__(self):
        self.called("__cuda_array_interface__")

    def __dlpack__(self, *args, **kwargs):
        self.called("__dlpack__")

    def __dlpack_device__(self):
        self.called("__dlpack_device__")

    def __iter__(self):
        self.called("__iter__")

    def __len__(self):
        self.called("__len__")

    def __index__(self):
        self.called("__index__")

    def __float__(self):
        self.called("__float__")


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class IsStorageReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("is_storage differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def non_storage_cases(self, module, conversion_trap):
        tensor = module.tensor([[1.0, 2.0], [3.0, 4.0]])
        return (
            tensor,
            tensor.transpose(0, 1),
            None,
            True,
            1,
            1.5,
            2.0j,
            "storage",
            b"storage",
            [],
            [1.0, 2.0],
            (1.0, 2.0),
            range(3),
            module.float32,
            module.device("cpu"),
            module.layout,
            module.strided,
            module.contiguous_format,
            module.preserve_format,
            inspect.getattr_static(module.Tensor, "dtype"),
            inspect.getattr_static(module.Tensor, "device"),
            module.Tensor,
            object(),
            conversion_trap,
        )

    def test_false_cases_and_conversion_hook_behavior_match_pytorch_2_13(self):
        actual_trap = ConversionTrap()
        expected_trap = ConversionTrap()
        actual_values = self.non_storage_cases(torch, actual_trap)
        expected_values = self.non_storage_cases(reference_torch, expected_trap)

        for case, (actual, expected) in enumerate(
            zip(actual_values, expected_values, strict=True)
        ):
            with self.subTest(case=case, value_type=type(actual).__name__):
                actual_result = torch.is_storage(actual)
                expected_result = reference_torch.is_storage(expected)
                self.assertIs(type(actual_result), type(expected_result))
                self.assertIs(actual_result, expected_result)
        self.assertEqual(actual_trap.calls, expected_trap.calls)
        self.assertEqual(actual_trap.calls, [])

    def test_callable_metadata_matches_pytorch_2_13(self):
        actual = torch.is_storage
        expected = reference_torch.is_storage

        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertIs(inspect.getmodule(actual), torch)
        self.assertIs(inspect.getmodule(expected), reference_torch)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(inspect.get_annotations(actual), inspect.get_annotations(expected))
        actual_hints = typing.get_type_hints(actual)
        expected_hints = typing.get_type_hints(expected)
        self.assertEqual(actual_hints.keys(), expected_hints.keys())
        self.assertIs(actual_hints["obj"], expected_hints["obj"])
        self.assertIs(
            typing.get_origin(actual_hints["return"]),
            typing.get_origin(expected_hints["return"]),
        )
        actual_union = typing.get_args(actual_hints["return"])[0]
        expected_union = typing.get_args(expected_hints["return"])[0]
        self.assertEqual(
            {storage_type.__name__ for storage_type in typing.get_args(actual_union)},
            {storage_type.__name__ for storage_type in typing.get_args(expected_union)},
        )
        self.assertEqual(
            {storage_type.__module__ for storage_type in typing.get_args(actual_union)},
            {storage_type.__module__ for storage_type in typing.get_args(expected_union)},
        )
        self.assertEqual(inspect.signature(actual), inspect.signature(expected))
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )

    def placement_contract(self, module):
        native = module._C
        direct_namespace = {}
        wildcard_namespace = {}
        native_wildcard_namespace = {}

        exec(f"from {module.__name__} import is_storage", direct_namespace)
        exec(f"from {module.__name__} import *", wildcard_namespace)
        exec(f"from {native.__name__} import *", native_wildcard_namespace)

        function = module.is_storage
        return (
            "is_storage" in vars(module),
            "is_storage" in vars(native),
            module.__all__.count("is_storage"),
            direct_namespace["is_storage"] is function,
            wildcard_namespace["is_storage"] is function,
            "is_storage" in native_wildcard_namespace,
        )

    def test_direct_and_wildcard_imports_match_pytorch_2_13(self):
        self.assertEqual(
            self.placement_contract(torch),
            self.placement_contract(reference_torch),
        )

    def test_copying_and_pickling_match_pytorch_2_13(self):
        actual = torch.is_storage
        expected = reference_torch.is_storage

        self.assertIs(copy.copy(actual), actual)
        self.assertIs(copy.copy(expected), expected)
        self.assertIs(copy.deepcopy(actual), actual)
        self.assertIs(copy.deepcopy(expected), expected)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(pickle.loads(pickle.dumps(expected, protocol)), expected)

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        cases = (
            (lambda: torch.is_storage(), lambda: reference_torch.is_storage()),
            (
                lambda: torch.is_storage(actual, actual),
                lambda: reference_torch.is_storage(expected, expected),
            ),
            (
                lambda: torch.is_storage(obj=actual),
                lambda: reference_torch.is_storage(obj=expected),
            ),
            (
                lambda: torch.is_storage(actual, obj=actual),
                lambda: reference_torch.is_storage(expected, obj=expected),
            ),
            (
                lambda: torch.is_storage(input=actual),
                lambda: reference_torch.is_storage(input=expected),
            ),
            (
                lambda: torch.is_storage(extra=actual),
                lambda: reference_torch.is_storage(extra=expected),
            ),
            (
                lambda: torch.is_storage(actual, extra=actual),
                lambda: reference_torch.is_storage(expected, extra=expected),
            ),
            (
                lambda: torch.is_storage(obj=actual, extra=actual),
                lambda: reference_torch.is_storage(obj=expected, extra=expected),
            ),
        )

        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_positive_reference_storage_cases_remain_unsupported_in_torch_rs(self):
        self.assertIs(
            reference_torch.is_storage(reference_torch.tensor([1.0]).untyped_storage()),
            True,
        )
        self.assertFalse(hasattr(torch, "Storage"))
        self.assertFalse(hasattr(torch, "TypedStorage"))
        self.assertFalse(hasattr(torch, "UntypedStorage"))
        self.assertFalse(hasattr(torch, "storage"))


if __name__ == "__main__":
    unittest.main()
