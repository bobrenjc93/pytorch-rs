import copy
import inspect
import pickle
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch

if __package__:
    from .signature_utils import assert_no_argument_signature
else:
    from signature_utils import assert_no_argument_signature


SIGMOID_DOC = """
sigmoid() -> Tensor

See :func:`torch.sigmoid`
"""

TOP_LEVEL_SIGMOID_DOC = """
sigmoid(input, *, out=None) -> Tensor

Alias for :func:`torch.special.expit`.
"""


SPECIAL_INPUT_BITS = np.asarray(
    (
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x8000_0001,
        0x007F_FFFF,
        0x807F_FFFF,
        0x0080_0000,
        0x8080_0000,
        0x3EFF_FFFF,
        0x3F00_0000,
        0x3F7F_FFFF,
        0x3F80_0000,
        0xBF00_0000,
        0xBF7F_FFFF,
        0xBF80_0000,
        0xBFC0_0000,
        0x3FC0_0000,
        0x42B0_0000,
        0x42B2_0000,
        0xC2B0_0000,
        0xC2B2_0000,
        0x7F7F_FFFF,
        0xFF7F_FFFF,
        0x7F80_0000,
        0xFF80_0000,
        0x7F81_2345,
        0xFF81_2345,
        0x7FC1_2345,
        0xFFC5_4321,
    ),
    dtype=np.uint32,
)

SPECIAL_OUTPUT_BITS = np.asarray(
    (
        0x3F00_0000,
        0x3F00_0000,
        0x3F00_0000,
        0x3F00_0000,
        0x3F00_0000,
        0x3F00_0000,
        0x3F00_0000,
        0x3F00_0000,
        0x3F1F_597F,
        0x3F1F_597F,
        0x3F3B_26A8,
        0x3F3B_26A8,
        0x3EC1_4D03,
        0x3E89_B2B1,
        0x3E89_B2B1,
        0x3E3A_CDC2,
        0x3F51_4C8F,
        0x3F80_0000,
        0x3F80_0000,
        0x0041_EDC4,
        0x0000_0000,
        0x3F80_0000,
        0x0000_0000,
        0x3F80_0000,
        0x0000_0000,
        0xFFC1_2345,
        0x7FC1_2345,
        0xFFC1_2345,
        0x7FC5_4321,
    ),
    dtype=np.uint32,
)


class TensorSigmoidTests(unittest.TestCase):
    @staticmethod
    def tensor_values(tensor):
        return np.asarray(tensor, dtype=np.float32)

    @staticmethod
    def tensor_bits(tensor):
        return np.asarray(tensor, dtype=np.float32).reshape(-1).view(np.uint32)

    def assert_result(self, actual, source, expected_stride, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, source.shape)
            self.assertEqual(actual.stride(), expected_stride)
            self.assertEqual(actual.storage_offset(), 0)
            self.assertFalse(actual.requires_grad)
            self.assertTrue(actual.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
            self.assertFalse(actual.is_set_to(source))
            if source.numel():
                self.assertNotEqual(actual.data_ptr(), source.data_ptr())

    @staticmethod
    def make_cases():
        base = torch.tensor(
            np.linspace(-3.75, 3.75, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist()
        )
        strided = base.transpose(0, 2)
        channels_last = torch.tensor(
            np.linspace(-15.0, 15.0, 120, dtype=np.float32)
            .reshape(2, 3, 4, 5)
            .tolist()
        ).contiguous(memory_format=torch.channels_last)
        channels_last_3d = torch.tensor(
            np.linspace(-90.0, 90.0, 720, dtype=np.float32)
            .reshape(2, 3, 4, 5, 6)
            .tolist()
        ).contiguous(memory_format=torch.channels_last_3d)
        return (
            ("scalar", torch.tensor(-0.0), ()),
            (
                "empty offset",
                torch.zeros((2, 0, 3)).transpose(0, 2)[1],
                (2, 1),
            ),
            ("empty singleton trailing", torch.zeros((0, 1)), (1, 1)),
            ("empty singleton middle", torch.zeros((0, 1, 2)), (2, 2, 1)),
            ("empty singleton surrounding", torch.zeros((1, 0, 1)), (1, 1, 1)),
            ("offset", strided[1], (1, 3)),
            ("noncontiguous", strided, (1, 4, 12)),
            ("channels last", channels_last, channels_last.stride()),
            (
                "channels last 3d",
                channels_last_3d,
                channels_last_3d.stride(),
            ),
            (
                "numerical edges",
                torch.tensor(memoryview(SPECIAL_INPUT_BITS.view(np.float32))),
                (1,),
            ),
        )

    @staticmethod
    def top_level_calls(source):
        return (
            ("positional", lambda: torch.sigmoid(source)),
            ("input", lambda: torch.sigmoid(input=source)),
            ("x", lambda: torch.sigmoid(x=source)),
            ("a", lambda: torch.sigmoid(a=source)),
            ("x1", lambda: torch.sigmoid(x1=source)),
            ("out none", lambda: torch.sigmoid(source, out=None)),
            ("alias and out none", lambda: torch.sigmoid(x=source, out=None)),
        )

    def test_values_layouts_offsets_empty_tensors_and_fresh_storage(self):
        smallest_subnormal = np.nextafter(np.float32(0), np.float32(1))
        for case, source, expected_stride in self.make_cases():
            output = source.sigmoid()
            self.assert_result(output, source, expected_stride, case=case)
            actual = self.tensor_values(output).reshape(-1)
            if case == "numerical edges":
                np.testing.assert_array_equal(actual.view(np.uint32), SPECIAL_OUTPUT_BITS)
            else:
                values = self.tensor_values(source).reshape(-1)
                with np.errstate(over="ignore", invalid="ignore"):
                    expected = np.float32(1.0) / (
                        np.float32(1.0) + np.exp(-values, dtype=np.float32)
                    )
                np.testing.assert_allclose(
                    actual,
                    expected,
                    rtol=2.0e-6,
                    atol=smallest_subnormal,
                    equal_nan=True,
                )

    def test_top_level_calls_reuse_tensor_sigmoid_bits_layouts_and_storage(self):
        for case, source, expected_stride in self.make_cases():
            expected = source.sigmoid()
            for form, call in self.top_level_calls(source):
                actual = call()
                self.assert_result(
                    actual, source, expected_stride, case=(case, form)
                )
                np.testing.assert_array_equal(
                    self.tensor_bits(actual), self.tensor_bits(expected)
                )

    @staticmethod
    def make_tracked_cases():
        scalar = torch.tensor(-1.25, requires_grad=True)
        empty = torch.zeros((2, 0, 3), requires_grad=True).transpose(0, 2)[1]
        leaf = torch.tensor(
            np.linspace(-3.75, 3.75, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist(),
            requires_grad=True,
        )
        strided = leaf.transpose(0, 2)
        return scalar, empty, strided[1], strided

    def test_active_autograd_is_rejected_before_planning(self):
        message = r"^sigmoid\(\): autograd recording is not supported$"
        for case, source in enumerate(self.make_tracked_cases()):
            with self.subTest(case=case):
                with self.assertRaisesRegex(RuntimeError, message):
                    source.sigmoid()

        extreme = torch.zeros((0,), requires_grad=True).reshape(
            (0, sys.maxsize, 3)
        )
        with self.assertRaisesRegex(RuntimeError, message):
            extreme.sigmoid()
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
                extreme.sigmoid()

    def test_detached_and_no_grad_inputs_use_the_inference_path(self):
        for case, source in enumerate(self.make_tracked_cases()):
            detached = source.detach()
            expected = detached.sigmoid()
            with torch.no_grad():
                actual = source.sigmoid()
            with self.subTest(case=case, mode="no_grad"):
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.stride(), expected.stride())
                self.assertEqual(actual.storage_offset(), expected.storage_offset())
                self.assertFalse(actual.requires_grad)
                self.assertTrue(actual.is_leaf)
                self.assertFalse(actual.is_set_to(source))
                np.testing.assert_array_equal(
                    self.tensor_values(actual).reshape(-1).view(np.uint32),
                    self.tensor_values(expected).reshape(-1).view(np.uint32),
                )
            with self.subTest(case=case, mode="detached"):
                self.assertFalse(expected.is_set_to(detached))
                if detached.numel():
                    self.assertNotEqual(expected.data_ptr(), detached.data_ptr())

    def test_tensorbase_descriptor_metadata_and_no_argument_errors(self):
        tensor = torch.tensor([1.25])
        descriptor = inspect.getattr_static(torch.Tensor, "sigmoid")
        bound = tensor.sigmoid

        self.assertIs(torch.Tensor.sigmoid, descriptor)
        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor), "<method 'sigmoid' of 'torch._C.TensorBase' objects>"
        )
        self.assertEqual(descriptor.__name__, "sigmoid")
        self.assertEqual(descriptor.__qualname__, "TensorBase.sigmoid")
        self.assertEqual(bound.__name__, "sigmoid")
        self.assertEqual(bound.__qualname__, "Tensor.sigmoid")
        self.assertEqual(descriptor.__doc__, SIGMOID_DOC)
        self.assertEqual(bound.__doc__, SIGMOID_DOC)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        assert_no_argument_signature(self, descriptor, "(self, /)")
        assert_no_argument_signature(self, bound, "()")

        cases = (
            (lambda: tensor.sigmoid(1), "TensorBase.sigmoid() takes no arguments (1 given)"),
            (lambda: bound(1), "Tensor.sigmoid() takes no arguments (1 given)"),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.sigmoid() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.sigmoid(1, 2),
                "TensorBase.sigmoid() takes no arguments (2 given)",
            ),
            (
                lambda: tensor.sigmoid(input=tensor),
                (
                    "Tensor.sigmoid() takes no keyword arguments"
                    if sys.version_info < (3, 11)
                    else "TensorBase.sigmoid() takes no keyword arguments"
                ),
            ),
            (lambda: bound(unexpected=True), "Tensor.sigmoid() takes no keyword arguments"),
            (
                lambda: descriptor(tensor, unexpected=True),
                "TensorBase.sigmoid() takes no keyword arguments",
            ),
            (lambda: descriptor(), "unbound method TensorBase.sigmoid() needs an argument"),
            (
                lambda: descriptor(1),
                "descriptor 'sigmoid' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.sigmoid() needs an argument",
            ),
        )
        for case, (call, message) in enumerate(cases):
            with self.subTest(case=case):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_torch_function_modes_dispatch_before_native_execution(self):
        tracked = torch.tensor([1.25], requires_grad=True)
        plain = tracked.detach()
        descriptor = inspect.getattr_static(torch.Tensor, "sigmoid")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode = RecordingMode(marker)
        with mode:
            result = tracked.sigmoid()
        self.assertIs(result, marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, (torch.Tensor,))
        self.assertEqual(len(args), 1)
        self.assertIs(args[0], tracked)
        self.assertIsNone(kwargs)

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = plain.sigmoid()
        self.assertEqual(order, ["upper", "lower"])
        np.testing.assert_allclose(forwarded.tolist(), [0.7772999], rtol=1.0e-6)

        order.clear()
        with self.assertRaisesRegex(
            RuntimeError, r"^sigmoid\(\): autograd recording is not supported$"
        ):
            with ForwardingMode("lower"):
                with ForwardingMode("upper"):
                    tracked.sigmoid()
        self.assertEqual(order, ["upper", "lower"])

        old_recursion_limit = sys.getrecursionlimit()
        declining = RecordingMode(NotImplemented)
        try:
            sys.setrecursionlimit(80)
            with declining:
                with self.assertRaises(RecursionError):
                    plain.sigmoid()
                self.assertEqual(
                    len(torch.overrides._get_current_function_mode_stack()), 1
                )
        finally:
            sys.setrecursionlimit(old_recursion_limit)
        self.assertGreater(len(declining.calls), 1)
        self.assertEqual(len(torch.overrides._get_current_function_mode_stack()), 0)

        invalid = RecordingMode(marker)
        with self.assertRaises(TypeError):
            with invalid:
                plain.sigmoid(1)
        self.assertEqual(invalid.calls, [])

    def test_top_level_autograd_rejection_and_no_grad_reuse_method_path(self):
        message = r"^sigmoid\(\): autograd recording is not supported$"
        for case, source in enumerate(self.make_tracked_cases()):
            detached = source.detach()
            expected = detached.sigmoid()
            for form, call in self.top_level_calls(source):
                with self.subTest(case=case, form=form, mode="recording"):
                    with self.assertRaisesRegex(RuntimeError, message):
                        call()
                with self.subTest(case=case, form=form, mode="no_grad"):
                    with torch.no_grad():
                        actual = call()
                    self.assertEqual(actual.shape, expected.shape)
                    self.assertEqual(actual.stride(), expected.stride())
                    self.assertEqual(
                        actual.storage_offset(), expected.storage_offset()
                    )
                    self.assertFalse(actual.requires_grad)
                    self.assertTrue(actual.is_leaf)
                    self.assertFalse(actual.is_set_to(source))
                    np.testing.assert_array_equal(
                        self.tensor_bits(actual), self.tensor_bits(expected)
                    )

        extreme = torch.zeros((0,), requires_grad=True).reshape(
            (0, sys.maxsize, 3)
        )
        with self.assertRaisesRegex(RuntimeError, message):
            torch.sigmoid(extreme)
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
                torch.sigmoid(extreme)

    def test_top_level_concrete_out_is_rejected_without_mutation(self):
        source = torch.tensor([1.25, -1.25], requires_grad=True)
        destination = torch.tensor([17.0, 19.0])
        destination_pointer = destination.data_ptr()
        expected_bits = self.tensor_bits(destination).copy()
        for form, call in (
            ("positional", lambda: torch.sigmoid(source, out=destination)),
            ("keyword", lambda: torch.sigmoid(input=source, out=destination)),
            ("alias", lambda: torch.sigmoid(x=source, out=destination)),
        ):
            with self.subTest(form=form):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^sigmoid\(\): the 'out' argument is not supported$",
                ):
                    call()
                self.assertEqual(destination.data_ptr(), destination_pointer)
                np.testing.assert_array_equal(
                    self.tensor_bits(destination), expected_bits
                )
                self.assertIsNone(source.grad)

    def test_top_level_modes_and_overrides_observe_original_call(self):
        tensor = torch.tensor([1.25], requires_grad=True)
        destination = torch.tensor([0.0])
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode = RecordingMode()
        with mode:
            self.assertIs(torch.sigmoid(input=tensor, out=destination), marker)
        self.assertEqual(
            mode.calls,
            [(torch.sigmoid, (), (), {"input": tensor, "out": destination})],
        )

        override_calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_calls.append((func, types, args, kwargs))
                return marker

        self.assertIs(torch.sigmoid(Override()), marker)
        self.assertIs(torch.sigmoid(torch.tensor([1.25]), out=Override()), marker)
        self.assertEqual(len(override_calls), 2)
        for function, dispatch_types, _, _ in override_calls:
            self.assertIs(function, torch.sigmoid)
            self.assertEqual(dispatch_types, (Override,))

        subclass_order = []

        class BaseOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_order.append("base")
                return marker

        class DerivedOverride(BaseOverride):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_order.append("derived")
                return marker

        self.assertIs(torch.sigmoid(BaseOverride(), out=DerivedOverride()), marker)
        self.assertEqual(subclass_order, ["derived"])

        forwarding_order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                forwarding_order.append(self.label)
                return func(*args, **(kwargs or {}))

        plain = tensor.detach()
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = torch.sigmoid(input=plain, out=None)
        self.assertEqual(forwarding_order, ["upper", "lower"])
        np.testing.assert_array_equal(
            self.tensor_bits(forwarded), self.tensor_bits(plain.sigmoid())
        )

        events = []

        class DecliningMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                events.append("mode")
                return NotImplemented

        class FallbackOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append("override")
                return marker

        with DecliningMode():
            self.assertIs(torch.sigmoid(FallbackOverride()), marker)
        self.assertEqual(events, ["mode", "override"])

        invalid = RecordingMode()
        with self.assertRaises(TypeError):
            with invalid:
                torch.sigmoid(tensor, tensor)
        self.assertEqual(invalid.calls, [])

    def test_top_level_builtin_metadata_exports_copying_and_pickling(self):
        function = torch.sigmoid
        self.assertIs(function, torch._C.sigmoid)
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "sigmoid")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.sigmoid")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, TOP_LEVEL_SIGMOID_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function), r"^<built-in method sigmoid of type object at 0x[0-9a-f]+>$"
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.sigmoid, function)
        for action in (
            lambda: setattr(owner, "sigmoid", None),
            lambda: delattr(owner, "sigmoid"),
        ):
            with self.assertRaises(TypeError):
                action()
            self.assertIs(owner.sigmoid, function)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("sigmoid"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["sigmoid"], function)

    def test_top_level_binding_and_type_error_precedence(self):
        tensor = torch.tensor([1.25])
        cases = (
            (
                lambda: torch.sigmoid(),
                'sigmoid() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.sigmoid(tensor, tensor),
                "sigmoid() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.sigmoid(tensor, input=tensor),
                "sigmoid() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.sigmoid(out=tensor),
                'sigmoid() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.sigmoid(extra=tensor),
                'sigmoid() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.sigmoid(1, extra=True),
                "sigmoid(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.sigmoid(input=[]),
                "sigmoid(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.sigmoid(tensor, out=[]),
                "sigmoid(): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.sigmoid(tensor, extra=True, out=[]),
                "sigmoid(): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.sigmoid(tensor, extra=True),
                "sigmoid() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.sigmoid(input=tensor, a=tensor),
                "sigmoid() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.sigmoid(a=tensor, x=tensor, out=None),
                "sigmoid() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.sigmoid(x=tensor, a=tensor, out=None),
                "sigmoid() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.sigmoid(np.zeros((2, 3), dtype=np.float32)),
                (
                    "sigmoid(): argument 'input' (position 1) must be Tensor, "
                    "not numpy.ndarray"
                ),
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

    def test_functional_and_inplace_forms_remain_unsupported(self):
        tensor = torch.tensor([1.25])
        self.assertTrue(hasattr(torch, "sigmoid"))
        self.assertFalse(hasattr(torch.nn.functional, "sigmoid"))
        self.assertFalse(hasattr(torch.Tensor, "sigmoid_"))
        self.assertFalse(hasattr(tensor, "sigmoid_"))
        self.assertFalse(hasattr(torch, "sigmoid_"))
        self.assertNotIn("sigmoid_", torch.__all__)
        with self.assertRaises(TypeError):
            tensor.sigmoid(out=None)


if __name__ == "__main__":
    unittest.main()
