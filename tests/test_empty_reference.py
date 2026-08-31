import copy
import pickle
import re
import sys
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
        self.assertEqual(
            actual_message.replace("torch.device or str", "torch.device"),
            expected_message,
        )

    def test_shape_and_keyword_forms_match_pytorch_2_13(self):
        cases = (
            ("scalar tuple", lambda module: module.empty(())),
            ("scalar list", lambda module: module.empty([])),
            ("single integer", lambda module: module.empty(2)),
            ("empty vector", lambda module: module.empty((0,))),
            ("empty middle", lambda module: module.empty([2, 0, 3])),
            ("multidimensional", lambda module: module.empty((2, 3))),
            ("size keyword", lambda module: module.empty(size=(2,))),
        )
        for case, create in cases:
            with self.subTest(case=case):
                actual = create(torch)
                expected = create(reference_torch)
                self.assertEqual(
                    self.tensor_contract(torch, actual),
                    self.tensor_contract(reference_torch, expected),
                )

    def test_integer_protocol_size_dimensions_match_pytorch_2_13(self):
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

    def test_dtype_device_layout_pin_and_requires_grad_match_pytorch_2_13(self):
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
            lambda module: {"requires_grad": None},
            lambda module: {"requires_grad": False},
            lambda module: {"requires_grad": True},
            lambda module: {"memory_format": None},
            lambda module: {"memory_format": module.contiguous_format},
            lambda module: module.nn.factory_kwargs(
                {"memory_format": module.contiguous_format}
            ),
            lambda module: {
                "out": None,
                "dtype": module.float32,
                "layout": module.strided,
                "device": module.device("cpu"),
                "pin_memory": False,
                "requires_grad": True,
                "memory_format": module.contiguous_format,
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
                    self.tensor_contract(torch, actual),
                    self.tensor_contract(reference_torch, expected),
                )

        actual = torch.empty(
            2,
            **torch.nn.factory_kwargs(
                {"memory_format": torch.contiguous_format}
            ),
        )
        expected = reference_torch.empty(
            2,
            **reference_torch.nn.factory_kwargs(
                {"memory_format": reference_torch.contiguous_format}
            ),
        )
        self.assertEqual(
            self.tensor_contract(torch, actual),
            self.tensor_contract(reference_torch, expected),
        )

    def test_empty_returns_fresh_storage_like_pytorch_2_13(self):
        def contract(module, shape):
            first = module.empty(shape)
            second = module.empty(shape)
            return {
                "first": self.tensor_contract(module, first),
                "second": self.tensor_contract(module, second),
                "fresh_view": not first.is_set_to(second),
                "fresh_data_ptr": (
                    None if first.numel() == 0 else first.data_ptr() != second.data_ptr()
                ),
            }

        for shape in ((), (2, 3), (2, 0, 3)):
            with self.subTest(shape=shape):
                self.assertEqual(contract(torch, shape), contract(reference_torch, shape))

    def test_error_order_matches_pytorch_2_13(self):
        cases = (
            ("missing size", lambda module: module.empty(out=None)),
            ("negative size", lambda module: module.empty(-1, out=None)),
            ("invalid dtype", lambda module: module.empty((1,), dtype=object(), out=None)),
            ("invalid device", lambda module: module.empty((1,), device=object(), out=None)),
            ("unknown keyword", lambda module: module.empty((1,), unexpected=True, out=None)),
            (
                "duplicate size",
                lambda module: module.empty((1,), size=(1,), out=None),
            ),
            (
                "invalid layout before negative size",
                lambda module: module.empty(-1, layout=object(), out=None),
            ),
            (
                "invalid pin before requires_grad",
                lambda module: module.empty(
                    (1,),
                    pin_memory=0,
                    requires_grad=0,
                    out=None,
                ),
            ),
        )
        for case, call in cases:
            with self.subTest(case=case):
                self.assert_error_matches(lambda: call(torch), lambda: call(reference_torch))

        overflow_cases = (
            lambda module: module.empty(2**63),
            lambda module: module.empty(np.uint64(2**63)),
            lambda module: module.empty(IndexDimension(2**63)),
            lambda module: module.empty((2**63, 0)),
        )
        for call in overflow_cases:
            with self.subTest(call=call):
                actual_type, actual_message = self.capture_error(lambda: call(torch))
                expected_type, expected_message = self.capture_error(
                    lambda: call(reference_torch)
                )
                self.assertIs(actual_type, expected_type)
                marker = "failed to unpack the object at pos 1 with error"
                self.assertIn(marker, actual_message)
                self.assertIn(marker, expected_message)
                self.assertIn("Overflow when unpacking long long", actual_message)
                self.assertIn("Overflow when unpacking long long", expected_message)

        sequence_boundary_cases = (
            lambda module: module.empty((True,)),
            lambda module: module.empty((np.bool_(True),)),
            lambda module: module.empty((-1,)),
            lambda module: module.empty((2, -1, 3)),
        )
        for call in sequence_boundary_cases:
            with self.subTest(call=call):
                self.assert_error_matches(lambda: call(torch), lambda: call(reference_torch))

        largest_zero_product = (2**63 - 1, 0)
        actual = torch.empty(largest_zero_product)
        expected = reference_torch.empty(largest_zero_product)
        self.assertEqual(
            self.tensor_contract(torch, actual),
            self.tensor_contract(reference_torch, expected),
        )

        storage_cases = (
            lambda module: module.empty((2**62, 4)),
            lambda module: module.empty((sys.maxsize // 4 + 1,)),
        )
        for call in storage_cases:
            with self.subTest(call=call):
                self.assert_error_matches(lambda: call(torch), lambda: call(reference_torch))

    def test_unsupported_dtype_device_layout_pin_out_and_empty_like_boundaries(self):
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
        meta = reference_torch.empty((1,), device="meta")
        self.assertEqual(str(meta.device), "meta")
        self.assertIs(meta.dtype, reference_torch.float32)
        self.assertIs(meta.layout, reference_torch.strided)

        for layout in (object(), reference_torch.sparse_coo):
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

        with self.assertRaisesRegex(
            TypeError,
            r"^empty\(\): argument 'memory_format' must be torch\.memory_format, not ",
        ):
            torch.empty((1,), memory_format=object())

        for memory_format in (
            torch.preserve_format,
            torch.channels_last,
            torch.channels_last_3d,
        ):
            with self.subTest(memory_format=memory_format):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^empty\(\): only torch\.contiguous_format memory_format is supported$",
                ):
                    torch.empty((1, 2, 3, 4, 5), memory_format=memory_format)

        out = torch.zeros((1,))
        with self.assertRaisesRegex(
            RuntimeError,
            r"^empty\(\): the 'out' argument is not supported$",
        ):
            torch.empty(
                (1,),
                out=out,
                layout=torch.strided,
                pin_memory=False,
            )
        self.assertEqual(out.tolist(), [0.0])

        self.assertFalse(hasattr(torch, "empty_like"))
        self.assertTrue(hasattr(reference_torch, "empty_like"))

    def mode_dispatch_observation(self, module):
        function = module.empty
        marker = object()
        intercepted = []

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append(
                    (
                        func is function,
                        types,
                        args,
                        kwargs,
                        len(module.overrides._get_current_function_mode_stack()),
                    )
                )
                return marker

        for call, expected_args, expected_kwargs in (
            (lambda: function((2, 3)), ((2, 3),), None),
            (lambda: function(size=(2, 3)), (), {"size": (2, 3)}),
        ):
            mode = RecordingMode()
            with mode:
                result = call()
                restored_inside = (
                    module.overrides._get_current_function_mode_stack() == [mode]
                )
            intercepted.append(
                (
                    result is marker,
                    mode.calls,
                    expected_args,
                    expected_kwargs,
                    restored_inside,
                    module.overrides._get_current_function_mode_stack() == [],
                )
            )

        forwarding_events = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                forwarding_events.append(
                    (
                        self.label,
                        tuple(
                            mode.label
                            for mode in module.overrides._get_current_function_mode_stack()
                        ),
                        func is function,
                        types,
                        args,
                        kwargs,
                    )
                )
                return func(*args, **(kwargs or {}))

        lower = ForwardingMode("lower")
        upper = ForwardingMode("upper")
        with lower:
            with upper:
                forwarded = function(size=(2, 3))
                nested_restored = (
                    module.overrides._get_current_function_mode_stack()
                    == [lower, upper]
                )
            lower_restored = (
                module.overrides._get_current_function_mode_stack() == [lower]
            )
        stack_empty = module.overrides._get_current_function_mode_stack() == []

        expected_error = ValueError("handler failed")

        class RaisingMode(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                raise expected_error

        raising = RaisingMode()
        with lower:
            with raising:
                try:
                    function((2, 3))
                except Exception as error:
                    handler_error = (
                        type(error).__name__,
                        str(error),
                        error.args,
                        error is expected_error,
                    )
                else:
                    handler_error = None
                handler_error_restored = (
                    module.overrides._get_current_function_mode_stack()
                    == [lower, raising]
                )
            handler_lower_restored = (
                module.overrides._get_current_function_mode_stack() == [lower]
            )

        class DecliningMode(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return NotImplemented

        declining = DecliningMode()
        with declining:
            try:
                function((2, 3))
            except Exception as error:
                declining_error = (
                    type(error).__name__,
                    re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
                    error.args[1:] if len(error.args) > 1 else (),
                )
            else:
                declining_error = None
            declining_restored = (
                module.overrides._get_current_function_mode_stack() == [declining]
            )

        return (
            intercepted,
            forwarding_events,
            self.tensor_contract(module, forwarded),
            nested_restored,
            lower_restored,
            stack_empty,
            handler_error,
            handler_error_restored,
            handler_lower_restored,
            declining_error,
            declining_restored,
            module.overrides._get_current_function_mode_stack() == [],
        )

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_dispatch_observation(torch),
            self.mode_dispatch_observation(reference_torch),
        )

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


if __name__ == "__main__":
    unittest.main()
