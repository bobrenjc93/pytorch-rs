import copy
import importlib
import inspect
import pickle
import re
import types
import unittest
import warnings

import numpy as np
import torch_rs as torch


FUNCTION_DOC_PREFIX = (
    "\nasarray(obj: Any, *, dtype: Optional[dtype], "
    "device: Optional[DeviceLikeType], copy: Optional[bool] = None, "
    "requires_grad: Optional[bool] = None) -> Tensor # noqa: B950\n\n"
    "Converts :attr:`obj` to a tensor."
)

REQUIRES_GRAD_WARNING = (
    "torch.asarray: unspecified requires_grad now defaults to obj.requires_grad "
    "instead of False. Pass requires_grad=False explicitly to get the old "
    "behavior and silence this warning."
)
UNSUPPORTED_ASARRAY_CONVERSION = (
    "asarray(): only exact native CPU float32 Tensor inputs, Python float scalars, "
    "or exact list/tuple sequences of Python floats are supported; NumPy "
    "arrays/scalars, integer and boolean inference, and other conversions are "
    "not implemented"
)
SEQUENCE_COPY_FALSE_ERROR = "can't alias arbitrary sequence into a tensor."


class AsArrayTests(unittest.TestCase):
    def assert_error(self, call, error_type, message):
        with self.assertRaisesRegex(error_type, f"^{re.escape(message)}$"):
            call()

    def tensor_cases(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=True
        )
        produced = leaf * 2.0
        tracked = produced.transpose(0, 1)
        source = torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ],
            dtype=torch.float32,
        )
        strided = source.transpose(0, 1)
        special_bits = np.asarray(
            (0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC12345),
            dtype=np.uint32,
        )
        return (
            ("scalar", torch.tensor(-0.0, dtype=torch.float32)),
            ("empty view", torch.zeros((2, 0, 3), dtype=torch.float32)[1]),
            ("strided view", strided[1]),
            ("leaf", leaf),
            ("tracked view", tracked),
            ("special bits", torch.tensor(memoryview(special_bits.view(np.float32)))),
        )

    def tensor_state(self, tensor):
        return (
            np.asarray(tensor).reshape(-1).view(np.uint32).tolist(),
            tensor.shape,
            tensor.stride(),
            tensor.storage_offset(),
            tensor.data_ptr(),
            tensor.dtype,
            tensor.device,
            tensor.layout,
            tensor.requires_grad,
            tensor.is_leaf,
            tensor.output_nr,
        )

    def float32_bits(self, tensor):
        return np.asarray(tensor).reshape(-1).view(np.uint32).tolist()

    def assert_sequence_result(
        self, data, expected_shape, expected_stride, expected_bits, **options
    ):
        result = torch.asarray(data, **options)
        duplicate = torch.asarray(data, **options)

        self.assertIsInstance(result, torch.Tensor)
        self.assertIsNot(result, duplicate)
        self.assertFalse(result.is_set_to(duplicate))
        if result.numel() > 0:
            self.assertNotEqual(result.data_ptr(), duplicate.data_ptr())
        self.assertEqual(result.shape, expected_shape)
        self.assertEqual(result.stride(), expected_stride)
        self.assertEqual(result.storage_offset(), 0)
        self.assertEqual(result.numel(), len(expected_bits))
        self.assertIs(result.dtype, torch.float32)
        self.assertEqual(result.device, torch.device("cpu"))
        self.assertIs(result.layout, torch.strided)
        self.assertFalse(result.requires_grad)
        self.assertTrue(result.is_leaf)
        self.assertEqual(result.output_nr, 0)
        self.assertIsNone(result.grad)
        self.assertEqual(self.float32_bits(result), expected_bits)

    def nested_singleton(self, value, depth, container):
        for _ in range(depth):
            value = [value] if container is list else (value,)
        return value

    def test_exact_native_cpu_float32_tensors_return_identical_object(self):
        option_cases = (
            {},
            {"dtype": None},
            {"dtype": torch.float32},
            {"dtype": torch.float},
            {"device": None},
            {"device": "cpu"},
            {"device": torch.device("cpu")},
            {"copy": None},
            {"copy": False},
            {"requires_grad": None},
            {
                "dtype": torch.float32,
                "device": torch.device("cpu"),
                "copy": False,
                "requires_grad": None,
            },
        )
        for case, tensor in self.tensor_cases():
            before = self.tensor_state(tensor)
            for options in option_cases:
                with self.subTest(case=case, options=options):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = torch.asarray(tensor, **options)
                    self.assertIs(result, tensor)
                    self.assertEqual(result.data_ptr(), before[4])
                    self.assertEqual(self.tensor_state(tensor), before)

    def test_python_float_scalars_create_fresh_cpu_float32_leaves(self):
        option_cases = (
            {},
            {"dtype": None},
            {"dtype": torch.float32},
            {"dtype": torch.float},
            {"device": None},
            {"device": "cpu"},
            {"device": torch.device("cpu")},
            {"copy": None},
            {"requires_grad": None},
            {
                "dtype": torch.float32,
                "device": torch.device("cpu"),
                "copy": None,
                "requires_grad": None,
            },
        )
        value_cases = (
            (0.0, 0x00000000),
            (-0.0, 0x80000000),
            (1.25, 0x3FA00000),
            (-3.5, 0xC0600000),
            (1e39, 0x7F800000),
            (-1e39, 0xFF800000),
            (float("inf"), 0x7F800000),
            (float("-inf"), 0xFF800000),
            (float("nan"), 0x7FC00000),
        )
        for value, expected_bits in value_cases:
            for options in option_cases:
                with self.subTest(value=repr(value), options=options):
                    result = torch.asarray(value, **options)
                    duplicate = torch.asarray(value, **options)

                    self.assertIsInstance(result, torch.Tensor)
                    self.assertIsNot(result, duplicate)
                    self.assertNotEqual(result.data_ptr(), duplicate.data_ptr())
                    self.assertFalse(result.is_set_to(duplicate))
                    self.assertEqual(result.shape, ())
                    self.assertEqual(result.stride(), ())
                    self.assertEqual(result.storage_offset(), 0)
                    self.assertEqual(result.numel(), 1)
                    self.assertIs(result.dtype, torch.float32)
                    self.assertEqual(result.device, torch.device("cpu"))
                    self.assertIs(result.layout, torch.strided)
                    self.assertFalse(result.requires_grad)
                    self.assertTrue(result.is_leaf)
                    self.assertEqual(result.output_nr, 0)
                    self.assertIsNone(result.grad)
                    self.assertEqual(self.float32_bits(result), [expected_bits])

    def test_python_float_sequences_create_fresh_cpu_float32_leaves(self):
        sequence_cases = (
            ("empty list", [], (0,), (1,), ()),
            ("empty tuple", (), (0,), (1,), ()),
            (
                "flat list",
                [1.25, -3.5],
                (2,),
                (1,),
                (0x3FA00000, 0xC0600000),
            ),
            (
                "flat tuple",
                (1.25, -3.5),
                (2,),
                (1,),
                (0x3FA00000, 0xC0600000),
            ),
            (
                "nested rectangular",
                [[1.0, -2.0], [3.5, 4.25]],
                (2, 2),
                (2, 1),
                (0x3F800000, 0xC0000000, 0x40600000, 0x40880000),
            ),
            (
                "mixed tuple/list",
                ([1.0, -0.0], [float("inf"), float("-inf")]),
                (2, 2),
                (2, 1),
                (0x3F800000, 0x80000000, 0x7F800000, 0xFF800000),
            ),
            (
                "special floats",
                [0.0, -0.0, float("nan"), float("inf"), float("-inf")],
                (5,),
                (1,),
                (0x00000000, 0x80000000, 0x7FC00000, 0x7F800000, 0xFF800000),
            ),
        )
        option_cases = (
            {},
            {"dtype": None},
            {"dtype": torch.float32},
            {"dtype": torch.float},
            {"device": None},
            {"device": "cpu"},
            {"device": torch.device("cpu")},
            {"copy": None},
            {"requires_grad": None},
            {
                "dtype": torch.float32,
                "device": torch.device("cpu"),
                "copy": None,
                "requires_grad": None,
            },
        )
        for case, data, expected_shape, expected_stride, expected_bits in sequence_cases:
            for options in option_cases:
                with self.subTest(case=case, options=options):
                    self.assert_sequence_result(
                        data,
                        expected_shape,
                        expected_stride,
                        list(expected_bits),
                        **options,
                    )

    def test_python_float_sequence_construction_ignores_no_grad(self):
        with torch.no_grad():
            result = torch.asarray([1.0, -0.0])

        self.assertFalse(result.requires_grad)
        self.assertTrue(result.is_leaf)
        self.assertEqual(result.output_nr, 0)
        self.assertEqual(self.float32_bits(result), [0x3F800000, 0x80000000])

    def test_recursive_and_overdeep_float_sequences_raise_value_error(self):
        recursive_list = []
        recursive_list.append(recursive_list)
        self.assert_error(
            lambda: torch.asarray(recursive_list),
            ValueError,
            "too many dimensions 'list'",
        )

        recursive_tuple = ([],)
        recursive_tuple[0].append(recursive_tuple)
        self.assert_error(
            lambda: torch.asarray(recursive_tuple),
            ValueError,
            "too many dimensions 'tuple'",
        )

        max_rank = self.nested_singleton(1.0, 128, list)
        result = torch.asarray(max_rank)
        self.assertEqual(result.shape, (1,) * 128)
        self.assertEqual(result.stride(), (1,) * 128)
        self.assertEqual(result.numel(), 1)
        self.assertEqual(float(result.reshape(())), 1.0)

        for container, type_name in ((list, "list"), (tuple, "tuple")):
            with self.subTest(container=type_name):
                overdeep = self.nested_singleton(1.0, 129, container)
                self.assert_error(
                    lambda: torch.asarray(overdeep),
                    ValueError,
                    f"too many dimensions '{type_name}'",
                )

    def test_copy_false_rejects_exact_sequences_before_conversion(self):
        recursive_list = []
        recursive_list.append(recursive_list)
        recursive_tuple = ([],)
        recursive_tuple[0].append(recursive_tuple)
        sequence_cases = (
            ("integer list", [1]),
            ("object list", [object()]),
            ("ragged list", [[1.0], [2.0, 3.0]]),
            ("recursive list", recursive_list),
            ("overdeep list", self.nested_singleton(1.0, 129, list)),
            ("object tuple", (object(),)),
            ("ragged tuple", ((1.0,), (2.0, 3.0))),
            ("recursive tuple", recursive_tuple),
            ("overdeep tuple", self.nested_singleton(1.0, 129, tuple)),
        )
        for case, data in sequence_cases:
            with self.subTest(case=case):
                self.assert_error(
                    lambda data=data: torch.asarray(data, copy=False),
                    ValueError,
                    SEQUENCE_COPY_FALSE_ERROR,
                )

    def test_identity_preserves_autograd_graph_and_gradient_object(self):
        leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        source = (leaf * 3.0).transpose(0, 1)[1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = torch.asarray(source, dtype=torch.float32, device="cpu")

        self.assertIs(result, source)
        self.assertFalse(result.is_leaf)
        self.assertEqual(result.output_nr, source.output_nr)

        result.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[0.0, 3.0, 0.0], [0.0, 3.0, 0.0]])
        gradient = leaf.grad
        self.assertIs(torch.asarray(leaf.grad), gradient)

    def test_requires_grad_identity_warning_matches_pytorch_message(self):
        previous_warn_always = torch.is_warn_always_enabled()
        torch.set_warn_always(True)
        try:
            leaf = torch.tensor([1.0], dtype=torch.float32, requires_grad=True)
            source = leaf * 2.0
            with self.assertWarnsRegex(UserWarning, re.escape(REQUIRES_GRAD_WARNING)):
                self.assertIs(torch.asarray(source), source)
        finally:
            torch.set_warn_always(previous_warn_always)

    def test_callable_metadata_exports_copy_pickle_and_reload(self):
        package = importlib.import_module("torch_rs")
        native = package._C
        function = package.asarray

        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "asarray")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.asarray")
        self.assertEqual(function.__module__, "torch")
        self.assertTrue(function.__doc__.startswith(FUNCTION_DOC_PREFIX))
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function),
            r"^<built-in method asarray of type object at 0x[0-9a-f]+>$",
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, package._C._VariableFunctionsClass)
        self.assertIs(owner.asarray, function)
        self.assertIs(native.asarray, function)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(package.__all__.count("asarray"), 1)
        self.assertNotIn("_VariableFunctionsClass", package.__all__)
        self.assertFalse(hasattr(package, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["asarray"], function)

        self.assertIs(importlib.reload(native), native)
        self.assertIs(native.asarray, function)
        self.assertIs(importlib.reload(package), package)
        self.assertIs(package.asarray, function)
        self.assertEqual(package.__all__.count("asarray"), 1)

    def test_torch_function_mode_dispatches_before_native_conversion(self):
        tensor = torch.tensor([1.0], dtype=torch.float32)
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        cases = (
            ("positional tensor", lambda: torch.asarray(tensor), (tensor,), None),
            (
                "keyword tensor",
                lambda: torch.asarray(obj=tensor, dtype=torch.float32),
                (),
                {"obj": tensor, "dtype": torch.float32},
            ),
            (
                "float sequence",
                lambda: torch.asarray([1.0, 2.0]),
                ([1.0, 2.0],),
                None,
            ),
            ("python float scalar", lambda: torch.asarray(1.25), (1.25,), None),
            (
                "unsupported device string",
                lambda: torch.asarray([1.0], device="cuda"),
                ([1.0],),
                {"device": "cuda"},
            ),
            (
                "copy request",
                lambda: torch.asarray(tensor, copy=True),
                (tensor,),
                {"copy": True},
            ),
            (
                "requires_grad request",
                lambda: torch.asarray(tensor, requires_grad=True),
                (tensor,),
                {"requires_grad": True},
            ),
        )
        for case, call, expected_args, expected_kwargs in cases:
            mode = RecordingMode()
            with mode:
                result = call()
            with self.subTest(case=case):
                self.assertIs(result, marker)
                self.assertEqual(len(mode.calls), 1)
                function, dispatch_types, args, kwargs = mode.calls[0]
                self.assertIs(function, torch.asarray)
                self.assertEqual(dispatch_types, ())
                self.assertEqual(args, expected_args)
                self.assertEqual(kwargs, expected_kwargs)

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    forwarded = torch.asarray(
                        obj=tensor, dtype=torch.float32, copy=False
                    )
        self.assertEqual(order, ["upper", "lower"])
        self.assertIs(forwarded, tensor)

        class DecliningMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaisesRegex(
            TypeError, r"^Multiple dispatch failed for 'torch\.asarray'"
        ):
            with DecliningMode():
                torch.asarray(tensor)
        self.assertEqual(len(torch.overrides._get_current_function_mode_stack()), 0)

    def test_binding_errors_and_unsupported_scope_are_explicit(self):
        tensor = torch.tensor([1.0], dtype=torch.float32)
        explicit_requires_grad = (
            "asarray(): explicit requires_grad changes are not supported; "
            "existing tensor autograd state is preserved"
        )
        cases = (
            (
                lambda: torch.asarray(),
                TypeError,
                'asarray() missing 1 required positional arguments: "obj"',
            ),
            (
                lambda: torch.asarray(tensor, tensor),
                TypeError,
                "asarray() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.asarray(tensor, obj=tensor),
                TypeError,
                "asarray() got multiple values for argument 'obj'",
            ),
            (
                lambda: torch.asarray(data=tensor),
                TypeError,
                'asarray() missing 1 required positional arguments: "obj"',
            ),
            (
                lambda: torch.asarray(tensor, data=tensor),
                TypeError,
                "asarray() got an unexpected keyword argument 'data'",
            ),
            (
                lambda: torch.asarray(tensor, out=None),
                TypeError,
                "asarray() got an unexpected keyword argument 'out'",
            ),
            (
                lambda: torch.asarray(tensor, pin_memory=False),
                TypeError,
                "asarray() got an unexpected keyword argument 'pin_memory'",
            ),
            (
                lambda: torch.asarray(tensor, dtype=1),
                TypeError,
                "asarray(): argument 'dtype' must be torch.dtype, not int",
            ),
            (
                lambda: torch.asarray(tensor, device=1.5),
                TypeError,
                "asarray(): argument 'device' must be torch.device, not float",
            ),
            (
                lambda: torch.asarray(tensor, copy=0),
                TypeError,
                "asarray(): argument 'copy' must be bool, not int",
            ),
            (
                lambda: torch.asarray(tensor, requires_grad=0),
                TypeError,
                "asarray(): argument 'requires_grad' must be bool, not int",
            ),
            (
                lambda: torch.asarray(tensor, device=""),
                RuntimeError,
                "Device string must not be empty",
            ),
            (
                lambda: torch.asarray(tensor, device="banana"),
                RuntimeError,
                "Expected one of cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, "
                "ideep, hip, ve, fpga, maia, xla, lazy, vulkan, mps, meta, hpu, "
                "mtia, privateuseone device type at start of device string: banana",
            ),
            (
                lambda: torch.asarray(tensor, device="cuda"),
                RuntimeError,
                "asarray(): device 'cuda' is not supported; only 'cpu' is implemented",
            ),
            (
                lambda: torch.asarray(1.0, device="meta"),
                RuntimeError,
                "asarray(): device 'meta' is not supported; only 'cpu' is implemented",
            ),
            (
                lambda: torch.asarray([1.0], device="cuda"),
                RuntimeError,
                "asarray(): device 'cuda' is not supported; only 'cpu' is implemented",
            ),
            (
                lambda: torch.asarray([1.0], device="meta"),
                RuntimeError,
                "asarray(): device 'meta' is not supported; only 'cpu' is implemented",
            ),
            (
                lambda: torch.asarray(tensor, device="cpu:0"),
                NotImplementedError,
                "asarray(): explicit indexed CPU devices require a copy and are not supported",
            ),
            (
                lambda: torch.asarray(1.0, device="cpu:0"),
                NotImplementedError,
                "asarray(): explicit indexed CPU devices require a copy and are not supported",
            ),
            (
                lambda: torch.asarray([1.0], device="cpu:0"),
                NotImplementedError,
                "asarray(): explicit indexed CPU devices require a copy and are not supported",
            ),
            (
                lambda: torch.asarray(tensor, device=torch.device("cpu", 1)),
                NotImplementedError,
                "asarray(): indexed CPU devices require a copy and are not supported",
            ),
            (
                lambda: torch.asarray([1.0], device=torch.device("cpu", 1)),
                NotImplementedError,
                "asarray(): indexed CPU devices require a copy and are not supported",
            ),
            (
                lambda: torch.asarray(tensor, copy=True),
                NotImplementedError,
                "asarray(): copy=True requires a copy and is not supported",
            ),
            (
                lambda: torch.asarray(1.0, copy=True),
                NotImplementedError,
                "asarray(): copy=True requires a copy and is not supported",
            ),
            (
                lambda: torch.asarray([1.0], copy=True),
                NotImplementedError,
                "asarray(): copy=True requires a copy and is not supported",
            ),
            (
                lambda: torch.asarray(1.0, copy=False),
                NotImplementedError,
                "asarray(): copy=False for Python float scalar inputs is not "
                "supported because scalar conversion requires fresh storage",
            ),
            (
                lambda: torch.asarray([1.0], copy=False),
                ValueError,
                SEQUENCE_COPY_FALSE_ERROR,
            ),
            (
                lambda: torch.asarray(tensor, requires_grad=False),
                NotImplementedError,
                explicit_requires_grad,
            ),
            (
                lambda: torch.asarray(tensor, requires_grad=True),
                NotImplementedError,
                explicit_requires_grad,
            ),
            (
                lambda: torch.asarray(1.0, requires_grad=False),
                NotImplementedError,
                explicit_requires_grad,
            ),
            (
                lambda: torch.asarray(1.0, requires_grad=True),
                NotImplementedError,
                explicit_requires_grad,
            ),
            (
                lambda: torch.asarray([1.0], requires_grad=False),
                NotImplementedError,
                explicit_requires_grad,
            ),
            (
                lambda: torch.asarray([1.0], requires_grad=True),
                NotImplementedError,
                explicit_requires_grad,
            ),
            (
                lambda: torch.asarray(np.asarray([1.0], dtype=np.float32)),
                NotImplementedError,
                UNSUPPORTED_ASARRAY_CONVERSION,
            ),
            (
                lambda: torch.asarray(np.float32(1.0)),
                NotImplementedError,
                UNSUPPORTED_ASARRAY_CONVERSION,
            ),
            (lambda: torch.asarray(1), NotImplementedError, UNSUPPORTED_ASARRAY_CONVERSION),
            (
                lambda: torch.asarray(True),
                NotImplementedError,
                UNSUPPORTED_ASARRAY_CONVERSION,
            ),
            (
                lambda: torch.asarray([1]),
                NotImplementedError,
                UNSUPPORTED_ASARRAY_CONVERSION,
            ),
            (
                lambda: torch.asarray([True]),
                NotImplementedError,
                UNSUPPORTED_ASARRAY_CONVERSION,
            ),
            (
                lambda: torch.asarray([np.float32(1.0)]),
                NotImplementedError,
                UNSUPPORTED_ASARRAY_CONVERSION,
            ),
            (
                lambda: torch.asarray([[1.0], (2,)]),
                NotImplementedError,
                UNSUPPORTED_ASARRAY_CONVERSION,
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message):
                self.assert_error(call, error_type, message)

        self.assert_error(
            lambda: torch.asarray([[1.0], [2.0, 3.0]]),
            ValueError,
            "expected a rectangular sequence, but nested shapes differ",
        )

        class ListSubclass(list):
            pass

        class TupleSubclass(tuple):
            pass

        self.assert_error(
            lambda: torch.asarray(ListSubclass([1.0])),
            NotImplementedError,
            UNSUPPORTED_ASARRAY_CONVERSION,
        )
        self.assert_error(
            lambda: torch.asarray(TupleSubclass((1.0,))),
            NotImplementedError,
            UNSUPPORTED_ASARRAY_CONVERSION,
        )

        class FloatSubclass(float):
            pass

        self.assert_error(
            lambda: torch.asarray(FloatSubclass(1.0)),
            NotImplementedError,
            UNSUPPORTED_ASARRAY_CONVERSION,
        )

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return object()

        self.assert_error(
            lambda: torch.asarray(Override()),
            NotImplementedError,
            UNSUPPORTED_ASARRAY_CONVERSION,
        )
        self.assertEqual(Override.calls, [])

    def test_error_ordering_still_validates_schema_before_sequence_conversion(self):
        recursive_list = []
        recursive_list.append(recursive_list)
        self.assert_error(
            lambda: torch.asarray([object()], dtype=1),
            TypeError,
            "asarray(): argument 'dtype' must be torch.dtype, not int",
        )
        self.assert_error(
            lambda: torch.asarray(recursive_list, dtype=1),
            TypeError,
            "asarray(): argument 'dtype' must be torch.dtype, not int",
        )
        self.assert_error(
            lambda: torch.asarray([object()], device=1.5),
            TypeError,
            "asarray(): argument 'device' must be torch.device, not float",
        )
        self.assert_error(
            lambda: torch.asarray([1.0], out=None),
            TypeError,
            "asarray() got an unexpected keyword argument 'out'",
        )
        self.assert_error(
            lambda: torch.asarray([1.0], requires_grad=True),
            NotImplementedError,
            "asarray(): explicit requires_grad changes are not supported; "
            "existing tensor autograd state is preserved",
        )

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return object()

        self.assert_error(
            lambda: torch.asarray(Override()),
            NotImplementedError,
            UNSUPPORTED_ASARRAY_CONVERSION,
        )
        self.assertEqual(Override.calls, [])


if __name__ == "__main__":
    unittest.main()
