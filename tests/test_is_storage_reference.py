import copy
import importlib
import inspect
import pickle
import unittest
import typing
import warnings

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
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def non_storage_cases(self, module, conversion_trap):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            requires_grad=True,
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        tracked.sum().backward()
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
            module.tensor(3.5),
            module.tensor([1.0, 2.0]),
            module.zeros((2, 0, 3)),
            module.zeros((2, 3, 4)).transpose(0, 2)[1],
            leaf,
            tracked,
            leaf.grad,
            module.float32,
            module.device("cpu"),
            module.layout,
            module.strided,
            module.memory_format,
            module.contiguous_format,
            inspect.getattr_static(module.Tensor, "dtype"),
            inspect.getattr_static(module.Tensor, "device"),
            module.Tensor,
            conversion_trap,
        )

    def test_non_storage_results_and_conversion_behavior_match_pytorch_2_13(self):
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
                self.assertIs(actual_result, False)
        self.assertEqual(actual_trap.calls, expected_trap.calls)
        self.assertEqual(actual_trap.calls, [])

    def callable_contract(self, module):
        function = module.is_storage
        resolved_annotations = typing.get_type_hints(function)
        storage_union = typing.get_args(resolved_annotations["return"])[0]
        storage_classes = typing.get_args(storage_union)
        direct_import = {}
        wildcard_import = {}
        exec(f"from {module.__name__} import is_storage", direct_import)
        exec(f"from {module.__name__} import *", wildcard_import)

        return {
            "type": type(function).__name__,
            "name": function.__name__,
            "qualname": function.__qualname__,
            "module": function.__module__.replace("torch_rs", "torch"),
            "getmodule": inspect.getmodule(function) is module,
            "doc": function.__doc__,
            "annotations": function.__annotations__,
            "resolved_obj": resolved_annotations["obj"] is typing.Any,
            "resolved_return_origin": str(
                typing.get_origin(resolved_annotations["return"])
            ).replace("torch_rs", "torch"),
            "resolved_storage_classes": tuple(
                (cls.__name__, cls.__module__.replace("torch_rs", "torch"))
                for cls in storage_classes
            ),
            "signature": str(inspect.signature(function)).replace(
                "torch_rs",
                "torch",
            ),
            "has_text_signature": hasattr(function, "__text_signature__"),
            "defaults": function.__defaults__,
            "kwdefaults": function.__kwdefaults__,
            "dict": function.__dict__,
            "all_count": module.__all__.count("is_storage"),
            "native_has_function": hasattr(module._C, "is_storage"),
            "direct_import": direct_import["is_storage"] is function,
            "wildcard_import": wildcard_import["is_storage"] is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_imports_copying_and_pickle_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )

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

    def test_reference_storage_positive_cases_remain_unsupported_locally(self):
        reference_tensor = reference_torch.tensor([1, 2, 3])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reference_storages = (
                reference_tensor.untyped_storage(),
                reference_torch.TypedStorage(5, dtype=reference_torch.float32),
            )
        for storage in reference_storages:
            with self.subTest(storage_type=type(storage).__name__):
                self.assertIs(reference_torch.is_storage(storage), True)

        for name in ("Storage", "TypedStorage", "UntypedStorage", "storage"):
            with self.subTest(name=name):
                self.assertTrue(hasattr(reference_torch, name))
                self.assertFalse(hasattr(torch, name))
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("torch_rs.storage")


if __name__ == "__main__":
    unittest.main()
