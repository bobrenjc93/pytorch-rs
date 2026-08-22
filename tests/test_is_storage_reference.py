import copy
import inspect
import pickle
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
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def non_storage_cases(self, module, conversion_trap):
        return (
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
            np.float32(1.0),
            np.array(1.0, dtype=np.float32),
            np.arange(4, dtype=np.float32),
            np.dtype(np.float32),
            module.tensor(3.5),
            module.zeros((2, 0, 3)),
            module.zeros((2, 3, 4)).transpose(0, 2)[1],
            module.float32,
            module.device("cpu"),
            module.strided,
            module.contiguous_format,
            module.preserve_format,
            inspect.getattr_static(module.Tensor, "dtype"),
            inspect.getattr_static(module.Tensor, "device"),
            module.Tensor,
            module.tensor,
            module,
            conversion_trap,
        )

    def test_false_results_and_conversion_hook_behavior_match_pytorch_2_13(self):
        actual_trap = ConversionTrap()
        expected_trap = ConversionTrap()
        actual_values = self.non_storage_cases(torch, actual_trap)
        expected_values = self.non_storage_cases(reference_torch, expected_trap)
        for case, (actual, expected) in enumerate(
            zip(actual_values, expected_values, strict=True)
        ):
            with self.subTest(case=case):
                actual_result = torch.is_storage(actual)
                expected_result = reference_torch.is_storage(expected)
                self.assertIs(type(actual_result), type(expected_result))
                self.assertIs(actual_result, expected_result)
        self.assertEqual(actual_trap.calls, expected_trap.calls)
        self.assertEqual(actual_trap.calls, [])

    def test_reference_storage_true_cases_without_exposing_storage_objects(self):
        tensor = reference_torch.tensor([1.0, 2.0, 3.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            storages = (
                reference_torch.UntypedStorage(3),
                reference_torch.TypedStorage(3, dtype=reference_torch.float32),
                reference_torch.FloatStorage(3),
                tensor.untyped_storage(),
                tensor.storage(),
            )

        for case, storage in enumerate(storages):
            with self.subTest(case=case, storage_type=type(storage).__name__):
                self.assertIs(reference_torch.is_storage(storage), True)
                self.assertIs(torch.is_storage(storage), False)

        for name in ("storage", "Storage", "TypedStorage", "UntypedStorage"):
            with self.subTest(owner="torch", name=name):
                self.assertFalse(hasattr(torch, name))
                self.assertTrue(hasattr(reference_torch, name))
        for name in ("storage", "storage_type", "untyped_storage", "_typed_storage"):
            with self.subTest(owner="Tensor", name=name):
                self.assertFalse(hasattr(torch.Tensor, name))
                self.assertTrue(hasattr(reference_torch.Tensor, name))

    def test_callable_metadata_exports_copying_and_pickling_match_pytorch_2_13(self):
        actual = torch.is_storage
        expected = reference_torch.is_storage
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(actual.__module__, "torch_rs")
        self.assertEqual(expected.__module__, "torch")
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(inspect.get_annotations(actual), inspect.get_annotations(expected))
        self.assertEqual(inspect.signature(actual), inspect.signature(expected))
        actual_hints = typing.get_type_hints(actual)
        expected_hints = typing.get_type_hints(expected)
        self.assertEqual(
            str(actual_hints).replace("torch_rs._is_storage", "torch.storage"),
            str(expected_hints),
        )
        self.assertIs(
            typing.get_origin(actual.__annotations__["return"]),
            typing.TypeGuard,
        )
        self.assertEqual(
            typing.get_args(actual.__annotations__["return"]),
            (typing.ForwardRef("TypedStorage | UntypedStorage"),),
        )
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )

        self.assertEqual(
            torch.__all__.count("is_storage"),
            reference_torch.__all__.count("is_storage"),
        )
        for module, function in ((torch, actual), (reference_torch, expected)):
            wildcard_namespace = {}
            exec(f"from {module.__name__} import *", wildcard_namespace)
            self.assertIs(wildcard_namespace["is_storage"], function)
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(module=module.__name__, protocol=protocol):
                    restored = pickle.loads(pickle.dumps(function, protocol=protocol))
                    self.assertIs(restored, function)

    def test_positional_only_binding_errors_match_pytorch_2_13(self):
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

        self.assertIs(
            torch.is_storage(actual, **{}),
            reference_torch.is_storage(expected, **{}),
        )


if __name__ == "__main__":
    unittest.main()
