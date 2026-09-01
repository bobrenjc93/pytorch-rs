import copy
import importlib
import pickle
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class IntSubclass(int):
    pass


class IndexDimension:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __index__(self):
        self.calls += 1
        return self.value


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class EmptyReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("empty differentials require pinned PyTorch 2.13.0")

    def tensor_contract(self, module, tensor):
        return {
            "shape": tuple(tensor.shape),
            "stride": tensor.stride(),
            "storage_offset": tensor.storage_offset(),
            "numel": tensor.numel(),
            "dtype": str(tensor.dtype),
            "dtype_identity": tensor.dtype is module.float32,
            "device": str(tensor.device),
            "layout": str(tensor.layout),
            "layout_identity": tensor.layout is module.strided,
            "is_pinned": tensor.is_pinned(),
            "requires_grad": tensor.requires_grad,
            "is_leaf": tensor.is_leaf,
            "grad_is_none": tensor.grad is None,
        }

    def capture_error(self, call):
        with self.assertRaises(Exception) as raised:
            call()
        return type(raised.exception), str(raised.exception)

    def assert_error_matches(self, actual_call, expected_call):
        actual_type, actual_message = self.capture_error(actual_call)
        expected_type, expected_message = self.capture_error(expected_call)
        self.assertIs(actual_type, expected_type)
        self.assertEqual(actual_message, expected_message)

    def test_shape_metadata_and_default_options_match_pytorch_2_13(self):
        cases = (
            ("single integer", lambda module: module.empty(2)),
            ("scalar tuple", lambda module: module.empty(())),
            ("scalar list", lambda module: module.empty([])),
            ("zero vector", lambda module: module.empty((0,))),
            ("zero middle", lambda module: module.empty([2, 0, 3])),
            ("multidimensional", lambda module: module.empty((2, 3, 1))),
            ("size keyword tuple", lambda module: module.empty(size=(2,))),
            ("size keyword list", lambda module: module.empty(size=[2])),
            ("out none", lambda module: module.empty((2,), out=None)),
            (
                "default layout",
                lambda module: module.empty((2,), layout=module.strided),
            ),
            ("default pin", lambda module: module.empty((2,), pin_memory=False)),
            ("dtype alias", lambda module: module.empty((2,), dtype=module.float)),
            ("cpu device string", lambda module: module.empty((2,), device="cpu")),
            (
                "cpu device object",
                lambda module: module.empty((2,), device=module.device("cpu")),
            ),
        )

        for case, create in cases:
            with self.subTest(case=case):
                actual = create(torch)
                expected = create(reference_torch)
                self.assertEqual(
                    self.tensor_contract(torch, actual),
                    self.tensor_contract(reference_torch, expected),
                )

    def test_integer_protocol_dimensions_match_pytorch_2_13(self):
        actual_single = IndexDimension(2)
        expected_single = IndexDimension(2)
        self.assertEqual(
            self.tensor_contract(torch, torch.empty(actual_single)),
            self.tensor_contract(reference_torch, reference_torch.empty(expected_single)),
        )
        self.assertEqual(actual_single.calls, 1)
        self.assertGreaterEqual(expected_single.calls, 1)

        actual_dynamic = IndexDimension(2)
        expected_dynamic = IndexDimension(2)
        actual = torch.empty([actual_dynamic, np.int64(3), IntSubclass(1)])
        expected = reference_torch.empty(
            [expected_dynamic, np.int64(3), IntSubclass(1)]
        )
        self.assertEqual(
            self.tensor_contract(torch, actual),
            self.tensor_contract(reference_torch, expected),
        )
        self.assertEqual(actual_dynamic.calls, 1)
        self.assertGreaterEqual(expected_dynamic.calls, 1)

    def test_requires_grad_and_no_grad_metadata_match_pytorch_2_13(self):
        for requires_grad in (None, False, True):
            with self.subTest(requires_grad=requires_grad):
                options = {"requires_grad": requires_grad}
                with torch.no_grad():
                    actual = torch.empty((2, 0, 3), **options)
                with reference_torch.no_grad():
                    expected = reference_torch.empty((2, 0, 3), **options)
                self.assertEqual(
                    self.tensor_contract(torch, actual),
                    self.tensor_contract(reference_torch, expected),
                )

    def test_empty_returns_fresh_storage_match_pytorch_2_13(self):
        def contract(module):
            first = module.empty((2, 3))
            second = module.empty((2, 3))
            scalar = module.empty(())
            scalar_peer = module.empty(())
            return {
                "first": self.tensor_contract(module, first),
                "second": self.tensor_contract(module, second),
                "fresh_pair_storage": first.data_ptr() != second.data_ptr(),
                "fresh_pair_view": not first.is_set_to(second),
                "fresh_scalar_storage": scalar.data_ptr() != scalar_peer.data_ptr(),
                "fresh_scalar_view": not scalar.is_set_to(scalar_peer),
            }

        self.assertEqual(contract(torch), contract(reference_torch))

    def test_size_validation_errors_match_pytorch_2_13(self):
        cases = (
            lambda module: module.empty(None),
            lambda module: module.empty(size=None),
            lambda module: module.empty(size=2),
            lambda module: module.empty(True),
            lambda module: module.empty([True]),
            lambda module: module.empty((1, -2)),
        )

        for call in cases:
            with self.subTest(call=call):
                self.assert_error_matches(
                    lambda: call(torch), lambda: call(reference_torch)
                )

    def test_rejects_unsupported_allocation_forms(self):
        with self.assertRaisesRegex(
            TypeError,
            r"^empty\(\) takes 1 positional argument but 2 were given$",
        ):
            torch.empty(2, 3)

        out = torch.empty((1,))
        with self.assertRaisesRegex(
            RuntimeError,
            r"^empty\(\): the 'out' argument is not supported$",
        ):
            torch.empty((1,), out=out)

        with self.assertRaisesRegex(
            TypeError,
            r"^empty\(\): argument 'dtype' must be torch\.dtype, not dtype$",
        ):
            torch.empty((1,), dtype=reference_torch.float64)

        with self.assertRaisesRegex(
            RuntimeError,
            r"^empty\(\): device 'meta' is not supported; only 'cpu' is implemented$",
        ):
            torch.empty((1,), device="meta")

        with self.assertRaisesRegex(
            TypeError,
            r"^empty\(\): argument 'layout' must be torch\.layout, not torch\.layout$",
        ):
            torch.empty((1,), layout=reference_torch.sparse_coo)

        with self.assertRaisesRegex(
            TypeError,
            r"^empty\(\): argument 'pin_memory' must be bool, not int$",
        ):
            torch.empty((1,), pin_memory=0)

        with self.assertRaisesRegex(
            RuntimeError,
            r"^empty\(\): pin_memory=True is not supported; only unpinned CPU storage is implemented$",
        ):
            torch.empty((1,), pin_memory=True)

    def test_rejects_subclass_and_mode_expansion(self):
        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return object()

        with self.assertRaises(TypeError):
            torch.empty(Override())
        self.assertEqual(Override.calls, [])

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return object()

        mode = RecordingMode()
        with mode:
            with self.assertRaisesRegex(
                NotImplementedError,
                r"^empty\(\): __torch_function__ modes are not supported$",
            ):
                torch.empty((1,))
        self.assertEqual(mode.calls, [])
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

    def test_callable_import_wildcard_reload_copy_and_pickle_behavior(self):
        package = importlib.import_module("torch_rs")
        native = package._C
        function = package.empty
        import_namespace = {}
        wildcard_namespace = {}
        exec("from torch_rs import empty as imported_empty", import_namespace)
        exec("from torch_rs import *", wildcard_namespace)

        self.assertEqual(
            {
                "callable": callable(function),
                "type": type(function).__name__,
                "name": function.__name__,
                "all_count": package.__all__.count("empty"),
                "owner_not_in_all": "_VariableFunctionsClass" not in package.__all__,
                "import_identity": import_namespace["imported_empty"] is function,
                "wildcard_identity": wildcard_namespace["empty"] is function,
                "copy_identity": copy.copy(function) is function,
                "deepcopy_identity": copy.deepcopy(function) is function,
                "pickle_identities": tuple(
                    pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                    for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
                ),
            },
            {
                "callable": callable(reference_torch.empty),
                "type": type(reference_torch.empty).__name__,
                "name": reference_torch.empty.__name__,
                "all_count": reference_torch.__all__.count("empty"),
                "owner_not_in_all": "_VariableFunctionsClass"
                not in reference_torch.__all__,
                "import_identity": True,
                "wildcard_identity": True,
                "copy_identity": copy.copy(reference_torch.empty) is reference_torch.empty,
                "deepcopy_identity": copy.deepcopy(reference_torch.empty)
                is reference_torch.empty,
                "pickle_identities": tuple(
                    pickle.loads(
                        pickle.dumps(reference_torch.empty, protocol=protocol)
                    )
                    is reference_torch.empty
                    for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
                ),
            },
        )

        self.assertIs(importlib.reload(native), native)
        self.assertIs(native.empty, function)
        self.assertIs(importlib.reload(package), package)
        self.assertIs(package.empty, function)
        self.assertEqual(package.__all__.count("empty"), 1)


if __name__ == "__main__":
    unittest.main()
