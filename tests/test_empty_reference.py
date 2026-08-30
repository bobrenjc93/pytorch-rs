import copy
import importlib
import inspect
import pickle
import re
import sys
import types
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
            "device_identity": tensor.device == module.device("cpu"),
            "layout": str(tensor.layout),
            "layout_identity": tensor.layout is module.strided,
            "requires_grad": tensor.requires_grad,
            "is_leaf": tensor.is_leaf,
            "grad_is_none": tensor.grad is None,
            "data_ptr_is_zero": tensor.data_ptr() == 0,
            "nbytes": tensor.nbytes,
            "is_pinned": tensor.is_pinned(),
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

    def test_zero_element_metadata_matches_pytorch_2_13(self):
        cases = (
            ("scalar zero", lambda module: module.empty(0)),
            ("tuple zero", lambda module: module.empty((0,))),
            ("list zero", lambda module: module.empty([0])),
            ("size object", lambda module: module.empty(module.Size([0]))),
            ("middle zero", lambda module: module.empty((2, 0, 3))),
            ("leading zero", lambda module: module.empty((0, 2))),
            ("sandwiched zero", lambda module: module.empty((1, 0, 1))),
            ("huge leading zero product", lambda module: module.empty((sys.maxsize, 0))),
            ("huge stride with leading zero", lambda module: module.empty((0, sys.maxsize, 1))),
            ("size keyword", lambda module: module.empty(size=(0,))),
            ("out none", lambda module: module.empty(size=(0,), out=None)),
            ("dtype none", lambda module: module.empty((0,), dtype=None)),
            ("dtype float32", lambda module: module.empty((0,), dtype=module.float32)),
            ("dtype float alias", lambda module: module.empty((0,), dtype=module.float)),
            ("device none", lambda module: module.empty((0,), device=None)),
            ("device string", lambda module: module.empty((0,), device="cpu")),
            ("device indexed string", lambda module: module.empty((0,), device="cpu:0")),
            ("device object", lambda module: module.empty((0,), device=module.device("cpu"))),
            (
                "indexed cpu device",
                lambda module: module.empty((0,), device=module.device("cpu", 2)),
            ),
            ("requires grad none", lambda module: module.empty((0,), requires_grad=None)),
            ("requires grad false", lambda module: module.empty((0,), requires_grad=False)),
            ("requires grad true", lambda module: module.empty((0,), requires_grad=True)),
            (
                "int subclasses",
                lambda module: module.empty((IntSubclass(0), np.uint32(3))),
            ),
        )
        for case, factory in cases:
            with self.subTest(case=case):
                actual = factory(torch)
                expected = factory(reference_torch)
                self.assertEqual(
                    self.tensor_contract(torch, actual),
                    self.tensor_contract(reference_torch, expected),
                )

    def test_index_protocol_dimensions_match_pytorch_2_13(self):
        actual_scalar = IndexDimension(0)
        expected_scalar = IndexDimension(0)
        actual_index = IndexDimension(0)
        expected_index = IndexDimension(0)

        actual = torch.empty(actual_scalar)
        expected = reference_torch.empty(expected_scalar)
        self.assertEqual(
            self.tensor_contract(torch, actual),
            self.tensor_contract(reference_torch, expected),
        )
        self.assertEqual(actual_scalar.calls, expected_scalar.calls)

        actual = torch.empty([actual_index, np.int64(2)])
        expected = reference_torch.empty([expected_index, np.int64(2)])
        self.assertEqual(
            self.tensor_contract(torch, actual),
            self.tensor_contract(reference_torch, expected),
        )
        self.assertEqual(actual_index.calls, expected_index.calls)

    def test_zero_element_storage_freshness_matches_pytorch_2_13(self):
        actual_first = torch.empty((2, 0, 3))
        actual_second = torch.empty((2, 0, 3))
        expected_first = reference_torch.empty((2, 0, 3))
        expected_second = reference_torch.empty((2, 0, 3))

        self.assertEqual(
            actual_first.is_set_to(actual_second),
            expected_first.is_set_to(expected_second),
        )
        self.assertEqual(actual_first.data_ptr(), expected_first.data_ptr())
        self.assertEqual(actual_second.data_ptr(), expected_second.data_ptr())

    def test_matching_shape_errors_match_pytorch_2_13(self):
        exact_cases = (
            lambda module: module.empty(-1),
            lambda module: module.empty(IndexDimension(-1)),
            lambda module: module.empty((-1,)),
            lambda module: module.empty((1, -2, 0)),
            lambda module: module.empty(True),
            lambda module: module.empty(False),
            lambda module: module.empty(np.bool_(True)),
            lambda module: module.empty((True,)),
            lambda module: module.empty((np.bool_(True),)),
            lambda module: module.empty(None),
            lambda module: module.empty(size=None),
            lambda module: module.empty(shape=(0,)),
            lambda module: module.empty(layout=module.strided),
        )
        for case in exact_cases:
            with self.subTest(case=case):
                self.assert_error_matches(lambda: case(torch), lambda: case(reference_torch))

        overflow_cases = (
            lambda module: module.empty(2**63),
            lambda module: module.empty(-(2**63) - 1),
            lambda module: module.empty((2**63,)),
            lambda module: module.empty((np.uint64(2**63),)),
            lambda module: module.empty(IndexDimension(2**63)),
        )
        for case in overflow_cases:
            with self.subTest(case=case):
                actual_type, actual_message = self.capture_error(lambda: case(torch))
                expected_type, expected_message = self.capture_error(
                    lambda: case(reference_torch)
                )
                self.assertIs(actual_type, expected_type)
                marker = "failed to unpack the object at pos 1 with error"
                self.assertIn(marker, actual_message)
                self.assertIn(marker, expected_message)
                self.assertIn("Overflow when unpacking long long", actual_message)
                self.assertIn("Overflow when unpacking long long", expected_message)

        self.assert_error_matches(
            lambda: torch.empty((0, 2**62, 4)),
            lambda: reference_torch.empty((0, 2**62, 4)),
        )
        self.assert_error_matches(
            lambda: torch.empty((sys.maxsize,)),
            lambda: reference_torch.empty((sys.maxsize,)),
        )

    def test_intentionally_unsupported_boundaries_remain_narrow(self):
        for size in ((), (1,), (2, 3), 1):
            with self.subTest(size=size):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    "nonzero-element uninitialized allocation is not supported",
                ):
                    torch.empty(size)

        for keyword, value in (
            ("shape", (0,)),
            ("layout", torch.strided),
            ("pin_memory", False),
            ("memory_format", torch.contiguous_format),
        ):
            with self.subTest(keyword=keyword):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"^empty\(\) got an unexpected keyword argument '{keyword}'$",
                ):
                    torch.empty((0,), **{keyword: value})

        with self.assertRaisesRegex(
            RuntimeError,
            r"^empty\(\): the 'out' argument is not supported$",
        ):
            torch.empty((0,), out=torch.zeros((0,)))
        with self.assertRaisesRegex(
            RuntimeError,
            r"^empty\(\): device 'meta' is not supported; only 'cpu' is implemented$",
        ):
            torch.empty((0,), device="meta")
        with self.assertRaisesRegex(
            TypeError,
            r"^empty\(\) takes 1 positional argument but 2 were given$",
        ):
            torch.empty(2, 0)
        self.assertFalse(hasattr(torch, "empty_like"))

    def callable_contract(self, module):
        function = module.empty
        owner = function.__reduce__()[1][0]
        wildcard_namespace = {}
        exec(f"from {module.__name__} import *", wildcard_namespace)
        try:
            inspect.signature(function)
        except Exception as error:
            signature_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
            )
        else:
            signature_error = None
        return {
            "type": type(function).__name__,
            "is_builtin": type(function) is types.BuiltinFunctionType,
            "name": function.__name__,
            "qualname": function.__qualname__,
            "module": function.__module__,
            "owner_name": owner.__name__,
            "owner_qualname": owner.__qualname__,
            "owner_module": owner.__module__.replace("torch_rs._C", "torch._C"),
            "owner_path_identity": owner is module._C._VariableFunctionsClass,
            "owner_callable_identity": owner.empty is function,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("empty"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["empty"] is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_imports_copy_pickle_and_reload_match_pytorch_2_13(self):
        self.assertEqual(self.callable_contract(torch), self.callable_contract(reference_torch))

        old = torch.empty
        native = torch._C
        self.assertIs(importlib.reload(native), native)
        self.assertIs(native.empty, old)
        self.assertIs(importlib.reload(torch), torch)
        self.assertIs(torch.empty, old)

    def normalize_dispatch_value(self, module, value):
        if value is module.float32:
            return "float32"
        if isinstance(value, tuple):
            return tuple(self.normalize_dispatch_value(module, item) for item in value)
        if isinstance(value, list):
            return [self.normalize_dispatch_value(module, item) for item in value]
        if isinstance(value, dict):
            return {
                key: self.normalize_dispatch_value(module, item)
                for key, item in value.items()
            }
        return value

    def mode_dispatch_observation(self, module, case):
        marker = object()
        testcase = self

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                stack = tuple(module.overrides._get_current_function_mode_stack())
                self.calls.append(
                    (
                        func.__name__,
                        func is module.empty,
                        types,
                        self.normalize_args(args),
                        self.normalize_kwargs(kwargs),
                        len(stack),
                    )
                )
                return marker

            def normalize_args(self, args):
                return tuple(
                    testcase.normalize_dispatch_value(module, arg)
                    for arg in args
                )

            def normalize_kwargs(self, kwargs):
                if kwargs is None:
                    return None
                return {
                    key: testcase.normalize_dispatch_value(module, value)
                    for key, value in kwargs.items()
                }

        mode = RecordingMode()
        with mode:
            if case == "positional":
                result = module.empty((0,), dtype=module.float32)
            else:
                result = module.empty(size=(0,), dtype=module.float32)
            stack_after_call = module.overrides._get_current_function_mode_stack()

        return {
            "result_is_marker": result is marker,
            "stack_after_call": (
                len(stack_after_call),
                stack_after_call == [mode],
            ),
            "stack_after_context": len(
                module.overrides._get_current_function_mode_stack()
            ),
            "calls": mode.calls,
        }

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        for case in ("positional", "keyword"):
            with self.subTest(case=case):
                self.assertEqual(
                    self.mode_dispatch_observation(torch, case),
                    self.mode_dispatch_observation(reference_torch, case),
                )


if __name__ == "__main__":
    unittest.main()
