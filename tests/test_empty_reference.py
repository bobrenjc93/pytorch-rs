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
            "has_data_ptr": tensor.data_ptr() != 0,
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

    def test_size_forms_and_metadata_match_pytorch_2_13(self):
        cases = (
            ("scalar tuple", lambda module: module.empty(())),
            ("scalar list", lambda module: module.empty([])),
            ("single integer", lambda module: module.empty(2)),
            ("empty vector", lambda module: module.empty((0,))),
            ("empty middle dimension", lambda module: module.empty([2, 0, 3])),
            ("multidimensional tuple", lambda module: module.empty((2, 3))),
            (
                "multidimensional size",
                lambda module: module.empty(module.Size([2, 3])),
            ),
            ("variadic", lambda module: module.empty(2, 3)),
            ("variadic empty", lambda module: module.empty(2, 0, 3)),
            ("size keyword", lambda module: module.empty(size=(2,))),
            ("dtype alias", lambda module: module.empty((2,), dtype=module.float)),
            ("layout none", lambda module: module.empty((2,), layout=None)),
            ("layout strided", lambda module: module.empty((2,), layout=module.strided)),
            ("device none", lambda module: module.empty((2,), device=None)),
            ("device string", lambda module: module.empty((2,), device="cpu")),
            (
                "device descriptor",
                lambda module: module.empty((2,), device=module.device("cpu")),
            ),
            (
                "indexed CPU descriptor",
                lambda module: module.empty((2,), device=module.device("cpu", 2)),
            ),
            ("pin memory false", lambda module: module.empty((2,), pin_memory=False)),
            ("requires grad false", lambda module: module.empty((2,), requires_grad=False)),
            ("requires grad true", lambda module: module.empty((2,), requires_grad=True)),
            (
                "all defaults explicit",
                lambda module: module.empty(
                    (2,),
                    out=None,
                    dtype=module.float32,
                    layout=module.strided,
                    device=module.device("cpu"),
                    pin_memory=False,
                    requires_grad=True,
                ),
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
        actual_sequence_dimension = IndexDimension(2)
        expected_sequence_dimension = IndexDimension(2)
        actual_sequence = torch.empty(
            [actual_sequence_dimension, np.int64(3), IntSubclass(1)]
        )
        expected_sequence = reference_torch.empty(
            [expected_sequence_dimension, np.int64(3), IntSubclass(1)]
        )
        self.assertEqual(
            self.tensor_contract(torch, actual_sequence),
            self.tensor_contract(reference_torch, expected_sequence),
        )
        self.assertEqual(actual_sequence_dimension.calls, 1)
        self.assertGreaterEqual(expected_sequence_dimension.calls, 1)

        actual_variadic_dimension = IndexDimension(2)
        expected_variadic_dimension = IndexDimension(2)
        actual_variadic = torch.empty(
            actual_variadic_dimension, np.int64(3), IntSubclass(1)
        )
        expected_variadic = reference_torch.empty(
            expected_variadic_dimension, np.int64(3), IntSubclass(1)
        )
        self.assertEqual(
            self.tensor_contract(torch, actual_variadic),
            self.tensor_contract(reference_torch, expected_variadic),
        )
        self.assertGreater(actual_variadic_dimension.calls, 0)
        self.assertGreaterEqual(expected_variadic_dimension.calls, 1)

    def test_requires_grad_no_grad_and_out_none_match_pytorch_2_13(self):
        cases = (
            ("default grad", lambda module: module.empty((2,))),
            ("requires grad", lambda module: module.empty((2,), requires_grad=True)),
            (
                "no grad default",
                lambda module: self.create_under_no_grad(module, requires_grad=False),
            ),
            (
                "no grad explicit requires grad",
                lambda module: self.create_under_no_grad(module, requires_grad=True),
            ),
            ("out none", lambda module: module.empty((2,), out=None)),
        )

        for case, create in cases:
            with self.subTest(case=case):
                actual = create(torch)
                expected = create(reference_torch)
                self.assertEqual(
                    self.tensor_contract(torch, actual),
                    self.tensor_contract(reference_torch, expected),
                )

    def create_under_no_grad(self, module, *, requires_grad):
        with module.no_grad():
            return module.empty((2,), requires_grad=requires_grad)

    def test_empty_returns_fresh_storage_like_pytorch_2_13(self):
        cases = (
            ("scalar", lambda module: module.empty(())),
            ("vector", lambda module: module.empty((2,))),
            ("empty", lambda module: module.empty((0,))),
            ("multidimensional", lambda module: module.empty((2, 3))),
            ("empty middle", lambda module: module.empty((2, 0, 3))),
        )

        for case, create in cases:
            with self.subTest(case=case):
                actual_first = create(torch)
                actual_second = create(torch)
                expected_first = create(reference_torch)
                expected_second = create(reference_torch)
                self.assertEqual(
                    actual_first.is_set_to(actual_second),
                    expected_first.is_set_to(expected_second),
                )
                self.assertEqual(
                    actual_first.data_ptr() == actual_second.data_ptr(),
                    expected_first.data_ptr() == expected_second.data_ptr(),
                )

    def test_error_order_for_supported_signature_matches_pytorch_2_13(self):
        cases = (
            ("missing size", lambda module: module.empty(out=None)),
            ("negative size", lambda module: module.empty((-1,), out=None)),
            (
                "invalid dtype",
                lambda module: module.empty((1,), dtype=object(), out=None),
            ),
            (
                "invalid device",
                lambda module: module.empty((1,), device=object(), out=None),
            ),
            (
                "unknown keyword",
                lambda module: module.empty((1,), unexpected=True, out=None),
            ),
            ("duplicate size", lambda module: module.empty((1,), size=(1,), out=None)),
            ("invalid layout", lambda module: module.empty((-1,), layout=object(), out=None)),
            ("invalid pin", lambda module: module.empty((-1,), pin_memory=0, out=None)),
            (
                "invalid pin before requires grad",
                lambda module: module.empty(
                    (1,), pin_memory=0, requires_grad=0, out=None
                ),
            ),
        )
        for case, call in cases:
            with self.subTest(case=case):
                actual_type, actual_message = self.capture_error(lambda: call(torch))
                expected_type, expected_message = self.capture_error(
                    lambda: call(reference_torch)
                )
                self.assertIs(actual_type, expected_type)
                self.assertEqual(
                    actual_message.replace("torch.device or str", "torch.device"),
                    expected_message,
                )

    def test_unsupported_options_are_rejected(self):
        self.assertFalse(hasattr(torch, "float64"))
        self.assertTrue(hasattr(reference_torch, "float64"))
        with self.assertRaisesRegex(
            TypeError,
            r"^empty\(\): argument 'dtype' must be torch\.dtype, not dtype$",
        ):
            torch.empty((1,), dtype=reference_torch.float64)
        self.assertIs(
            reference_torch.empty((1,), dtype=reference_torch.float64).dtype,
            reference_torch.float64,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            r"^empty\(\): device 'meta' is not supported; only 'cpu' is implemented$",
        ):
            torch.empty((1,), device="meta")
        self.assertEqual(
            str(reference_torch.empty((1,), device="meta").device), "meta"
        )

        for layout in (object(), reference_torch.strided, reference_torch.sparse_coo):
            with self.subTest(layout=layout):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^empty\(\): argument 'layout' must be torch\.layout, not ",
                ):
                    torch.empty((1,), layout=layout)
        self.assertIs(
            reference_torch.empty((1,), layout=reference_torch.strided).layout,
            reference_torch.strided,
        )
        self.assertIs(
            reference_torch.empty((1,), layout=reference_torch.sparse_coo).layout,
            reference_torch.sparse_coo,
        )

        for pin_memory in (0, 1, "false", object()):
            with self.subTest(pin_memory=pin_memory):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^empty\(\): argument 'pin_memory' must be bool, not ",
                ):
                    torch.empty((1,), pin_memory=pin_memory)
        with self.assertRaisesRegex(
            RuntimeError,
            r"^empty\(\): pin_memory=True is not supported; only unpinned CPU storage is implemented$",
        ):
            torch.empty((1,), pin_memory=True)

        out = torch.zeros((1,))
        with self.assertRaisesRegex(
            RuntimeError,
            r"^empty\(\): the 'out' argument is not supported$",
        ):
            torch.empty((1,), out=out)
        self.assertEqual(out.tolist(), [0.0])
        reference_out = reference_torch.zeros((1,))
        self.assertIs(reference_torch.empty((1,), out=reference_out), reference_out)

        with self.assertRaisesRegex(
            RuntimeError,
            r"^empty\(\): memory_format is not supported; only the default contiguous memory format is implemented$",
        ):
            torch.empty((1,), memory_format=torch.contiguous_format)
        with self.assertRaisesRegex(
            RuntimeError,
            r"^empty\(\): memory_format is not supported; only the default contiguous memory format is implemented$",
        ):
            torch.empty((1,), memory_format=None)

        self.assertFalse(hasattr(torch, "empty_like"))
        self.assertTrue(hasattr(reference_torch, "empty_like"))

    def test_callable_import_wildcard_copy_and_pickle_match_pytorch_2_13(self):
        def contract(module):
            function = module.empty
            import_namespace = {}
            wildcard_namespace = {}
            exec(
                f"from {module.__name__} import empty as imported_empty",
                import_namespace,
            )
            exec(f"from {module.__name__} import *", wildcard_namespace)
            return {
                "callable": callable(function),
                "type": type(function).__name__,
                "name": function.__name__,
                "all_count": module.__all__.count("empty"),
                "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
                "import_identity": import_namespace["imported_empty"] is function,
                "wildcard_identity": wildcard_namespace["empty"] is function,
                "copy_identity": copy.copy(function) is function,
                "deepcopy_identity": copy.deepcopy(function) is function,
                "pickle_identities": tuple(
                    pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                    for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
                ),
            }

        self.assertEqual(contract(torch), contract(reference_torch))

    def test_reload_preserves_empty_export_identity(self):
        package = importlib.import_module("torch_rs")
        native = package._C
        function = package.empty

        self.assertIs(native.empty, function)
        self.assertIs(importlib.reload(native), native)
        self.assertIs(native.empty, function)
        self.assertIs(importlib.reload(package), package)
        self.assertIs(package.empty, function)
        self.assertEqual(package.__all__.count("empty"), 1)


if __name__ == "__main__":
    unittest.main()
