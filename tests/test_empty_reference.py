import copy
import ctypes
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

    def tensor_metadata_contract(self, module, tensor):
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

    def test_shape_and_metadata_match_pytorch_2_13_without_value_contract(self):
        cases = (
            ("scalar tuple", lambda module: module.empty(())),
            ("scalar list", lambda module: module.empty([])),
            ("single int", lambda module: module.empty(2)),
            ("empty middle", lambda module: module.empty([2, 0, 3])),
            ("multidimensional", lambda module: module.empty((2, 3))),
            ("keyword size", lambda module: module.empty(size=[2])),
            (
                "explicit float alias",
                lambda module: module.empty((2,), dtype=module.float),
            ),
        )
        for case, create in cases:
            with self.subTest(case=case):
                actual = create(torch)
                expected = create(reference_torch)
                self.assertEqual(
                    self.tensor_metadata_contract(torch, actual),
                    self.tensor_metadata_contract(reference_torch, expected),
                )

    def test_out_none_layout_pin_and_requires_grad_match_pytorch_2_13(self):
        option_factories = (
            lambda module: {},
            lambda module: {"dtype": None},
            lambda module: {"dtype": module.float32},
            lambda module: {"dtype": module.float},
            lambda module: {"out": None},
            lambda module: {"layout": None},
            lambda module: {"layout": module.strided},
            lambda module: {"device": None},
            lambda module: {"device": "cpu"},
            lambda module: {"device": "cpu:0"},
            lambda module: {"device": module.device("cpu")},
            lambda module: {"device": module.device("cpu", 2)},
            lambda module: {"pin_memory": None},
            lambda module: {"pin_memory": False},
            lambda module: {"memory_format": None},
            lambda module: {"memory_format": module.contiguous_format},
            lambda module: {"requires_grad": None},
            lambda module: {"requires_grad": False},
            lambda module: {"requires_grad": True},
            lambda module: {
                "out": None,
                "dtype": module.float32,
                "layout": module.strided,
                "device": module.device("cpu"),
                "pin_memory": False,
                "memory_format": module.contiguous_format,
                "requires_grad": True,
            },
        )
        for option_factory in option_factories:
            actual_options = option_factory(torch)
            expected_options = option_factory(reference_torch)
            with self.subTest(options=actual_options):
                with torch.no_grad():
                    actual = torch.empty((2, 3), **actual_options)
                with reference_torch.no_grad():
                    expected = reference_torch.empty((2, 3), **expected_options)
                self.assertEqual(
                    self.tensor_metadata_contract(torch, actual),
                    self.tensor_metadata_contract(reference_torch, expected),
                )

    def test_fresh_storage_matches_pytorch_2_13_without_value_contract(self):
        def contract(module, size):
            first = module.empty(size, out=None)
            second = module.empty(size, out=None)
            return {
                "first": self.tensor_metadata_contract(module, first),
                "second": self.tensor_metadata_contract(module, second),
                "same_data_ptr": first.data_ptr() == second.data_ptr(),
                "same_view": first.is_set_to(second),
            }

        for size in ((), (2,), (2, 0, 3), (2, 3)):
            with self.subTest(size=size):
                self.assertEqual(
                    contract(torch, size),
                    contract(reference_torch, size),
                )

    def test_data_ptr_writes_match_pytorch_2_13(self):
        values = (12.5, -3.0, 7.25)

        def contract(module):
            tensor = module.empty((len(values),), dtype=module.float32)
            pointer = tensor.data_ptr()
            self.assertNotEqual(pointer, 0)
            buffer = (ctypes.c_float * len(values)).from_address(pointer)
            buffer[:] = values
            return tensor.data_ptr() == pointer, tensor.tolist()

        self.assertEqual(contract(torch), contract(reference_torch))

    def test_integer_protocol_single_size_matches_pytorch_2_13(self):
        for value_factory in (
            lambda: IntSubclass(2),
            lambda: np.int64(2),
            lambda: np.uint32(2),
            lambda: IndexDimension(2),
        ):
            actual_value = value_factory()
            expected_value = value_factory()
            with self.subTest(value=type(actual_value).__name__):
                actual = torch.empty(actual_value)
                expected = reference_torch.empty(expected_value)
                self.assertEqual(
                    self.tensor_metadata_contract(torch, actual),
                    self.tensor_metadata_contract(reference_torch, expected),
                )
            if isinstance(actual_value, IndexDimension):
                self.assertEqual(actual_value.calls, 1)
                self.assertGreaterEqual(expected_value.calls, 1)

    def test_invalid_sequence_dimensions_match_pytorch_2_13_errors(self):
        def exception_contract(module, create):
            try:
                create(module)
            except Exception as error:
                return type(error), str(error).split("\n", 1)[0]
            self.fail("expected the size sequence to be rejected")

        exact_cases = (
            lambda module: module.empty([True]),
            lambda module: module.empty([False]),
            lambda module: module.empty((True,)),
            lambda module: module.empty([np.bool_(True)]),
            lambda module: module.empty([-1]),
            lambda module: module.empty([1, -2]),
            lambda module: module.empty([np.int64(-1)]),
        )
        for create in exact_cases:
            with self.subTest(create=create):
                self.assertEqual(
                    exception_contract(torch, create),
                    exception_contract(reference_torch, create),
                )

        overflow_cases = (
            lambda module: module.empty([2**63]),
            lambda module: module.empty([-(2**63) - 1]),
            lambda module: module.empty([np.uint64(2**63)]),
            lambda module: module.empty([IndexDimension(2**63)]),
        )
        for create in overflow_cases:
            with self.subTest(create=create):
                actual_type, actual_message = exception_contract(torch, create)
                expected_type, expected_message = exception_contract(
                    reference_torch,
                    create,
                )
                self.assertIs(actual_type, expected_type)
                self.assertIn(
                    "argument 'size' failed to unpack the object at pos 1",
                    actual_message,
                )
                self.assertIn("Overflow when unpacking long long", actual_message)
                self.assertIn(
                    "argument 'size' failed to unpack the object at pos 1",
                    expected_message,
                )
                self.assertIn("Overflow when unpacking long long", expected_message)

    def test_error_boundaries_match_or_pin_narrow_unsupported_surface(self):
        for dtype in (reference_torch.float64,):
            with self.subTest(dtype=dtype):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^empty\(\): argument 'dtype' must be torch\.dtype, not dtype$",
                ):
                    torch.empty((1,), dtype=dtype)
                self.assertIs(reference_torch.empty((1,), dtype=dtype).dtype, dtype)

        with self.assertRaisesRegex(
            RuntimeError,
            r"^empty\(\): device 'meta' is not supported; only 'cpu' is implemented$",
        ):
            torch.empty((1,), device="meta")
        self.assertEqual(
            str(reference_torch.empty((1,), device="meta").device),
            "meta",
        )

        for layout in (object(), reference_torch.strided, reference_torch.sparse_coo):
            with self.subTest(layout=layout):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^empty\(\): argument 'layout' must be torch\.layout, not ",
                ):
                    torch.empty((1,), layout=layout)

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

        for memory_format in (0, "contiguous", object()):
            with self.subTest(memory_format=memory_format):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^empty\(\): argument 'memory_format' must be torch\.memory_format, not ",
                ):
                    torch.empty((1,), memory_format=memory_format)
                with self.assertRaisesRegex(
                    TypeError,
                    r"^empty\(\): argument 'memory_format' must be torch\.memory_format, not ",
                ):
                    reference_torch.empty((1,), memory_format=memory_format)

        for memory_format in (
            torch.preserve_format,
            torch.channels_last,
            torch.channels_last_3d,
        ):
            with self.subTest(memory_format=memory_format):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    r"^empty\(\): only contiguous memory_format is supported$",
                ):
                    torch.empty((2, 3), memory_format=memory_format)

        out = torch.zeros((1,))
        with self.assertRaisesRegex(
            RuntimeError,
            r"^empty\(\): the 'out' argument is not supported$",
        ):
            torch.empty((1,), out=out, layout=torch.strided, pin_memory=False)

        self.assertTrue(hasattr(reference_torch, "empty_like"))
        self.assertFalse(hasattr(torch, "empty_like"))
        with self.assertRaises(AttributeError):
            torch.empty_like(torch.empty((1,)))

    def test_callable_import_and_wildcard_exports_match_pytorch_2_13(self):
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


if __name__ == "__main__":
    unittest.main()
