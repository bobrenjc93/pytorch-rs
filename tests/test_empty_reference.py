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

    def __index__(self):
        return self.value


class StatefulIndexDimension:
    def __init__(self, values):
        self.values = values
        self.calls = 0

    def __index__(self):
        value = self.values[min(self.calls, len(self.values) - 1)]
        self.calls += 1
        return value


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class EmptyReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("empty differentials require pinned PyTorch 2.13.0")

    def tensor_observation(self, module, tensor):
        return (
            tuple(tensor.shape),
            tensor.stride(),
            tensor.storage_offset(),
            str(tensor.dtype),
            tensor.dtype is module.float32,
            str(tensor.device),
            str(tensor.layout),
            tensor.layout is module.strided,
            tensor.is_pinned(),
            tensor.requires_grad,
            tensor.is_leaf,
            tensor.is_contiguous(),
        )

    def capture_error(self, call):
        with self.assertRaises(Exception) as raised:
            call()
        return type(raised.exception), str(raised.exception)

    def test_scalar_results_and_metadata_match_pytorch_2_13(self):
        dimension_factories = (
            lambda: 2,
            lambda: 0,
            lambda: IntSubclass(2),
            lambda: np.int64(2),
            lambda: np.uint32(2),
            lambda: IndexDimension(2),
        )
        metadata_factories = (
            lambda module: {},
            lambda module: {"out": None},
            lambda module: {"dtype": module.float32},
            lambda module: {"layout": None},
            lambda module: {"layout": module.strided},
            lambda module: {"device": "cpu"},
            lambda module: {"device": module.device("cpu")},
            lambda module: {"pin_memory": None},
            lambda module: {"pin_memory": False},
            lambda module: {"memory_format": None},
            lambda module: {"memory_format": module.contiguous_format},
            lambda module: {
                "out": None,
                "dtype": module.float32,
                "layout": module.strided,
                "device": module.device("cpu"),
                "pin_memory": False,
                "requires_grad": True,
            },
        )

        for dimension_factory in dimension_factories:
            for metadata_factory in metadata_factories:
                actual_dimension = dimension_factory()
                expected_dimension = dimension_factory()
                actual_keywords = metadata_factory(torch)
                expected_keywords = metadata_factory(reference_torch)
                with self.subTest(
                    dimension=actual_dimension,
                    keywords=actual_keywords,
                ):
                    actual = torch.empty(actual_dimension, **actual_keywords)
                    expected = reference_torch.empty(
                        expected_dimension, **expected_keywords
                    )
                    self.assertEqual(
                        self.tensor_observation(torch, actual),
                        self.tensor_observation(reference_torch, expected),
                    )

    def test_empty_and_variadic_results_and_metadata_match_pytorch_2_13(self):
        dimension_factories = (
            (lambda: 2, lambda: 3),
            (lambda: 2, lambda: 0, lambda: 3),
            (lambda: 2, lambda: False),
            (lambda: 2, lambda: True),
            (
                lambda: IntSubclass(2),
                lambda: np.int64(3),
                lambda: np.uint32(1),
                lambda: IndexDimension(2),
            ),
        )
        metadata_factories = (
            lambda module: {},
            lambda module: {"out": None},
            lambda module: {"dtype": module.float32},
            lambda module: {"layout": None},
            lambda module: {"layout": module.strided},
            lambda module: {"device": "cpu"},
            lambda module: {"device": module.device("cpu")},
            lambda module: {"pin_memory": None},
            lambda module: {"pin_memory": False},
            lambda module: {"memory_format": None},
            lambda module: {"memory_format": module.contiguous_format},
            lambda module: {
                "out": None,
                "dtype": module.float32,
                "layout": module.strided,
                "device": module.device("cpu"),
                "pin_memory": False,
                "requires_grad": True,
            },
        )

        for dimension_factory in dimension_factories:
            for metadata_factory in metadata_factories:
                actual_dimensions = tuple(factory() for factory in dimension_factory)
                expected_dimensions = tuple(factory() for factory in dimension_factory)
                actual_keywords = metadata_factory(torch)
                expected_keywords = metadata_factory(reference_torch)
                with self.subTest(
                    dimensions=actual_dimensions,
                    keywords=actual_keywords,
                ):
                    actual = torch.empty(*actual_dimensions, **actual_keywords)
                    expected = reference_torch.empty(
                        *expected_dimensions, **expected_keywords
                    )
                    self.assertEqual(
                        self.tensor_observation(torch, actual),
                        self.tensor_observation(reference_torch, expected),
                    )

    def test_tuple_list_and_size_keyword_match_pytorch_2_13(self):
        cases = (
            ("scalar tuple", lambda module: module.empty(())),
            ("scalar list", lambda module: module.empty([])),
            ("empty tuple", lambda module: module.empty((0,))),
            ("tuple", lambda module: module.empty((2, 3))),
            ("list", lambda module: module.empty([2, 3])),
            ("tuple non-leading bool", lambda module: module.empty((2, True))),
            ("list non-leading bool", lambda module: module.empty([2, False])),
            ("size keyword", lambda module: module.empty(size=(2,))),
            ("size keyword non-leading bool", lambda module: module.empty(size=(2, True))),
            (
                "integer protocol dimensions",
                lambda module: module.empty(
                    [IndexDimension(2), np.int64(3), IntSubclass(1)]
                ),
            ),
        )
        for case, factory in cases:
            with self.subTest(case=case):
                actual = factory(torch)
                expected = factory(reference_torch)
                self.assertEqual(
                    self.tensor_observation(torch, actual),
                    self.tensor_observation(reference_torch, expected),
                )

        actual_kwargs = torch.nn.factory_kwargs(
            {"memory_format": torch.contiguous_format}
        )
        expected_kwargs = reference_torch.nn.factory_kwargs(
            {"memory_format": reference_torch.contiguous_format}
        )
        actual = torch.empty(2, **actual_kwargs)
        expected = reference_torch.empty(2, **expected_kwargs)
        self.assertEqual(
            self.tensor_observation(torch, actual),
            self.tensor_observation(reference_torch, expected),
        )

    def test_variadic_leading_index_provider_calls_match_pytorch_2_13(self):
        actual_dimension = StatefulIndexDimension((2, 3, 4))
        expected_dimension = StatefulIndexDimension((2, 3, 4))

        actual = torch.empty(actual_dimension, 3)
        expected = reference_torch.empty(expected_dimension, 3)

        self.assertEqual(
            self.tensor_observation(torch, actual),
            self.tensor_observation(reference_torch, expected),
        )
        self.assertEqual(actual_dimension.calls, expected_dimension.calls)

    def test_out_none_results_and_storage_freshness_match_pytorch_2_13(self):
        cases = (
            ("scalar", lambda module: module.empty(2, out=None)),
            ("variadic", lambda module: module.empty(2, 3, out=None)),
            ("variadic empty", lambda module: module.empty(2, 0, 3, out=None)),
            ("tuple", lambda module: module.empty((2, 3), out=None)),
            ("list", lambda module: module.empty([2, 3], out=None)),
            ("size keyword", lambda module: module.empty(size=(2,), out=None)),
            (
                "requires grad",
                lambda module: module.empty(2, 3, out=None, requires_grad=True),
            ),
            ("empty", lambda module: module.empty((0,), out=None)),
            ("scalar tensor", lambda module: module.empty((), out=None)),
        )

        for case, factory in cases:
            with self.subTest(case=case):
                actual = factory(torch)
                actual_peer = factory(torch)
                expected = factory(reference_torch)
                expected_peer = factory(reference_torch)
                self.assertEqual(
                    self.tensor_observation(torch, actual),
                    self.tensor_observation(reference_torch, expected),
                )
                self.assertEqual(
                    actual.is_set_to(actual_peer),
                    expected.is_set_to(expected_peer),
                )
                self.assertEqual(
                    actual.data_ptr() == actual_peer.data_ptr(),
                    expected.data_ptr() == expected_peer.data_ptr(),
                )

    def test_no_grad_requires_grad_behavior_matches_pytorch_2_13(self):
        with torch.no_grad():
            actual_default = torch.empty((2, 3), dtype=torch.float32)
            actual_tracked = torch.empty(
                (2, 3), dtype=torch.float32, requires_grad=True
            )
        with reference_torch.no_grad():
            expected_default = reference_torch.empty(
                (2, 3), dtype=reference_torch.float32
            )
            expected_tracked = reference_torch.empty(
                (2, 3), dtype=reference_torch.float32, requires_grad=True
            )

        self.assertEqual(
            self.tensor_observation(torch, actual_default),
            self.tensor_observation(reference_torch, expected_default),
        )
        self.assertEqual(
            self.tensor_observation(torch, actual_tracked),
            self.tensor_observation(reference_torch, expected_tracked),
        )

    def mode_dispatch_observation(self, module):
        function = module.empty
        marker = object()
        negative_size = [-1]
        overflow_size = [2**63, 0]
        intercepted = []

        def normalize(value):
            for name in (
                "contiguous_format",
                "preserve_format",
                "channels_last",
                "channels_last_3d",
            ):
                if hasattr(module, name) and value is getattr(module, name):
                    return f"torch.{name}"
            if isinstance(value, tuple):
                return tuple(normalize(item) for item in value)
            if isinstance(value, list):
                return [normalize(item) for item in value]
            if isinstance(value, dict):
                return {key: normalize(item) for key, item in value.items()}
            return value

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append(
                    (
                        func is function,
                        types,
                        normalize(args),
                        normalize(kwargs),
                        len(module.overrides._get_current_function_mode_stack()),
                    )
                )
                return marker

        cases = (
            (lambda: function(2), (2,), None),
            (lambda: function(negative_size), (negative_size,), None),
            (lambda: function(overflow_size), (overflow_size,), None),
            (lambda: function(2, device="cuda"), (2,), {"device": "cuda"}),
            (lambda: function(2, pin_memory=True), (2,), {"pin_memory": True}),
            (lambda: function(2, memory_format=None), (2,), {"memory_format": None}),
            (
                lambda: function(2, memory_format=module.contiguous_format),
                (2,),
                {"memory_format": module.contiguous_format},
            ),
            (
                lambda: function(2, memory_format=module.channels_last),
                (2,),
                {"memory_format": module.channels_last},
            ),
            (lambda: function(2, requires_grad=True), (2,), {"requires_grad": True}),
        )
        for call, expected_args, expected_kwargs in cases:
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
                    normalize(expected_args),
                    normalize(expected_kwargs),
                    restored_inside,
                    module.overrides._get_current_function_mode_stack() == [],
                )
            )

        predispatch_errors = []
        for call in (
            lambda: function([True, 2]),
            lambda: function(size=(True,)),
            lambda: function(size=2),
            lambda: function(1.2),
            lambda: function(2, dtype=object()),
            lambda: function(2, requires_grad=1),
            lambda: function(2, memory_format=object()),
        ):
            mode = RecordingMode()
            with mode:
                try:
                    call()
                except Exception as error:
                    error_result = (type(error).__name__, str(error), error.args)
                else:
                    error_result = None
                restored_inside = (
                    module.overrides._get_current_function_mode_stack() == [mode]
                )
            predispatch_errors.append(
                (
                    error_result,
                    mode.calls,
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
                        normalize(args),
                        normalize(kwargs),
                    )
                )
                return func(*args, **(kwargs or {}))

        lower = ForwardingMode("lower")
        upper = ForwardingMode("upper")
        with lower:
            with upper:
                forwarded = function(2, True)
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
                    function(2)
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

        native_error_mode = ForwardingMode("native-error")
        with native_error_mode:
            try:
                function([True, 2])
            except Exception as error:
                native_error = (type(error).__name__, str(error), error.args)
            else:
                native_error = None
            native_error_restored = (
                module.overrides._get_current_function_mode_stack()
                == [native_error_mode]
            )

        class DecliningMode(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return NotImplemented

        declining = DecliningMode()
        with declining:
            try:
                function(2)
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
            predispatch_errors,
            forwarding_events,
            self.tensor_observation(module, forwarded),
            nested_restored,
            lower_restored,
            stack_empty,
            handler_error,
            handler_error_restored,
            handler_lower_restored,
            native_error,
            native_error_restored,
            declining_error,
            declining_restored,
            module.overrides._get_current_function_mode_stack() == [],
        )

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_dispatch_observation(torch),
            self.mode_dispatch_observation(reference_torch),
        )

    def test_dimension_errors_match_pytorch_2_13(self):
        exact_cases = (
            -1,
            IndexDimension(-1),
            True,
            False,
            np.bool_(True),
            sys.maxsize,
            IndexDimension(sys.maxsize),
        )
        for dimension in exact_cases:
            with self.subTest(dimension=dimension):
                actual_type, actual_message = self.capture_error(
                    lambda dimension=dimension: torch.empty(dimension)
                )
                expected_type, expected_message = self.capture_error(
                    lambda dimension=dimension: reference_torch.empty(dimension)
                )
                self.assertIs(actual_type, expected_type)
                self.assertEqual(actual_message, expected_message)

        overflow_cases = (
            2**63,
            -(2**63) - 1,
            np.uint64(2**63),
            IndexDimension(2**63),
        )
        for dimension in overflow_cases:
            with self.subTest(dimension=dimension):
                actual_type, actual_message = self.capture_error(
                    lambda dimension=dimension: torch.empty(dimension)
                )
                expected_type, expected_message = self.capture_error(
                    lambda dimension=dimension: reference_torch.empty(dimension)
                )
                self.assertIs(actual_type, expected_type)
                marker = "failed to unpack the object at pos 1 with error"
                self.assertIn(marker, actual_message)
                self.assertIn(marker, expected_message)
                self.assertIn("Overflow when unpacking long long", actual_message)
                self.assertIn("Overflow when unpacking long long", expected_message)

        variadic_exact_cases = (
            (2, -1),
            (2, IndexDimension(-1)),
            (2, np.bool_(True)),
            (sys.maxsize, 2),
        )
        for dimensions in variadic_exact_cases:
            with self.subTest(dimensions=dimensions):
                actual_type, actual_message = self.capture_error(
                    lambda dimensions=dimensions: torch.empty(*dimensions)
                )
                expected_type, expected_message = self.capture_error(
                    lambda dimensions=dimensions: reference_torch.empty(*dimensions)
                )
                self.assertIs(actual_type, expected_type)
                self.assertEqual(actual_message, expected_message)

        variadic_overflow_cases = (
            (2, 2**63),
            (2, np.uint64(2**63)),
            (2, IndexDimension(2**63)),
        )
        for dimensions in variadic_overflow_cases:
            with self.subTest(dimensions=dimensions):
                actual_type, actual_message = self.capture_error(
                    lambda dimensions=dimensions: torch.empty(*dimensions)
                )
                expected_type, expected_message = self.capture_error(
                    lambda dimensions=dimensions: reference_torch.empty(*dimensions)
                )
                self.assertIs(actual_type, expected_type)
                marker = "failed to unpack the object at pos 2 with error"
                self.assertIn(marker, actual_message)
                self.assertIn(marker, expected_message)
                self.assertIn("Overflow when unpacking long long", actual_message)
                self.assertIn("Overflow when unpacking long long", expected_message)

        sequence_exact_factories = (
            lambda: (True, 2),
            lambda: [True, 2],
            lambda: (np.bool_(True), 2),
            lambda: [np.bool_(True), 2],
            lambda: (-1,),
            lambda: [-1],
            lambda: (2, -1),
            lambda: [2, -1],
            lambda: (2, IndexDimension(-1)),
            lambda: [2, IndexDimension(-1)],
        )
        for size_factory in sequence_exact_factories:
            actual_size = size_factory()
            expected_size = size_factory()
            with self.subTest(size=actual_size):
                actual_type, actual_message = self.capture_error(
                    lambda size=actual_size: torch.empty(size)
                )
                expected_type, expected_message = self.capture_error(
                    lambda size=expected_size: reference_torch.empty(size)
                )
                self.assertIs(actual_type, expected_type)
                self.assertEqual(actual_message, expected_message)

        keyword_size_exact_factories = (
            lambda: {"size": 2},
            lambda: {"size": 1.2},
            lambda: {"size": (True,)},
            lambda: {"size": [True]},
        )
        for kwargs_factory in keyword_size_exact_factories:
            actual_kwargs = kwargs_factory()
            expected_kwargs = kwargs_factory()
            with self.subTest(kwargs=actual_kwargs):
                actual_type, actual_message = self.capture_error(
                    lambda kwargs=actual_kwargs: torch.empty(**kwargs)
                )
                expected_type, expected_message = self.capture_error(
                    lambda kwargs=expected_kwargs: reference_torch.empty(**kwargs)
                )
                self.assertIs(actual_type, expected_type)
                self.assertEqual(actual_message, expected_message)

        sequence_overflow_cases = (
            (lambda: (2**63, 0), 1),
            (lambda: [2**63, 0], 1),
            (lambda: (2, np.uint64(2**63)), 2),
            (lambda: [2, IndexDimension(2**63)], 2),
        )
        for size_factory, position in sequence_overflow_cases:
            actual_size = size_factory()
            expected_size = size_factory()
            with self.subTest(size=actual_size):
                actual_type, actual_message = self.capture_error(
                    lambda size=actual_size: torch.empty(size)
                )
                expected_type, expected_message = self.capture_error(
                    lambda size=expected_size: reference_torch.empty(size)
                )
                self.assertIs(actual_type, expected_type)
                marker = f"failed to unpack the object at pos {position} with error"
                self.assertIn(marker, actual_message)
                self.assertIn(marker, expected_message)
                self.assertIn("Overflow when unpacking long long", actual_message)
                self.assertIn("Overflow when unpacking long long", expected_message)

    def test_unsupported_dtype_device_layout_pin_memory_and_memory_format_boundaries(
        self,
    ):
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

        for actual_memory_format, expected_memory_format in (
            (None, None),
            (torch.contiguous_format, reference_torch.contiguous_format),
        ):
            with self.subTest(memory_format=actual_memory_format):
                actual = torch.empty((1,), memory_format=actual_memory_format)
                expected = reference_torch.empty(
                    (1,), memory_format=expected_memory_format
                )
                self.assertEqual(
                    self.tensor_observation(torch, actual),
                    self.tensor_observation(reference_torch, expected),
                )

        actual_type, actual_message = self.capture_error(
            lambda: torch.empty((1,), memory_format=object())
        )
        expected_type, expected_message = self.capture_error(
            lambda: reference_torch.empty((1,), memory_format=object())
        )
        self.assertIs(actual_type, expected_type)
        self.assertEqual(actual_message, expected_message)

        with self.assertRaisesRegex(
            TypeError,
            r"^empty\(\): argument 'memory_format' must be torch\.memory_format, not ",
        ):
            torch.empty((1,), memory_format=reference_torch.contiguous_format)

        for memory_format in (
            torch.preserve_format,
            torch.channels_last,
            torch.channels_last_3d,
        ):
            with self.subTest(memory_format=memory_format):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    r"^empty\(\): only default-equivalent memory_format is supported$",
                ):
                    torch.empty((1,), memory_format=memory_format)

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

        signature = (
            "empty(*size, *, out=None, dtype=None, layout=torch.strided, "
            "device=None, requires_grad=False, pin_memory=False, "
            "memory_format=torch.contiguous_format) -> Tensor"
        )
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
            "doc_signature": signature in (function.__doc__ or ""),
            "text_signature": function.__text_signature__,
            "signature_error": signature_error,
            "all_count": module.__all__.count("empty"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "wildcard_identity": wildcard_namespace["empty"] is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_exports_copy_pickle_and_reload_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )

        old = torch.empty
        native = torch._C
        self.assertIs(importlib.reload(native), native)
        self.assertIs(native.empty, old)
        self.assertIs(importlib.reload(torch), torch)
        self.assertIs(torch.empty, old)


if __name__ == "__main__":
    unittest.main()
