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


TRUNC_DOC = """
trunc() -> Tensor

See :func:`torch.trunc`
"""

TOP_LEVEL_TRUNC_DOC = """
trunc(input, *, out=None) -> Tensor

Returns a new tensor with the truncated integer values of
the elements of :attr:`input`.

For integer inputs, follows the array-api convention of returning a
copy of the input tensor.

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 3.4742,  0.5466, -0.8008, -0.9079])
    >>> torch.trunc(a)
    tensor([ 3.,  0., -0., -0.])
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
        0x0000_0000,
        0x8000_0000,
        0x0000_0000,
        0x8000_0000,
        0x0000_0000,
        0x8000_0000,
        0x0000_0000,
        0x8000_0000,
        0x0000_0000,
        0x0000_0000,
        0x0000_0000,
        0x3F80_0000,
        0x8000_0000,
        0x8000_0000,
        0xBF80_0000,
        0xBF80_0000,
        0x3F80_0000,
        0x7F7F_FFFF,
        0xFF7F_FFFF,
        0x7F80_0000,
        0xFF80_0000,
        0x7FC1_2345,
        0xFFC1_2345,
        0x7FC1_2345,
        0xFFC5_4321,
    ),
    dtype=np.uint32,
)


def make_cases(module):
    base = module.tensor(
        np.linspace(-3.75, 3.75, 24, dtype=np.float32)
        .reshape(2, 3, 4)
        .tolist(),
        dtype=module.float32,
    )
    strided = base.transpose(0, 2)
    channels_last = module.tensor(
        np.linspace(-15.0, 15.0, 120, dtype=np.float32)
        .reshape(2, 3, 4, 5)
        .tolist(),
        dtype=module.float32,
    ).contiguous(memory_format=module.channels_last)
    channels_last_3d = module.tensor(
        np.linspace(-90.0, 90.0, 720, dtype=np.float32)
        .reshape(2, 3, 4, 5, 6)
        .tolist(),
        dtype=module.float32,
    ).contiguous(memory_format=module.channels_last_3d)
    return (
        ("scalar", module.tensor(-0.0, dtype=module.float32), ()),
        (
            "empty offset",
            module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
            (2, 1),
        ),
        ("empty singleton trailing", module.zeros((0, 1)), (1, 1)),
        ("empty singleton middle", module.zeros((0, 1, 2)), (2, 2, 1)),
        ("empty singleton surrounding", module.zeros((1, 0, 1)), (1, 1, 1)),
        ("offset", strided[1], (1, 3)),
        ("noncontiguous", strided, (1, 4, 12)),
        ("channels last", channels_last, channels_last.stride()),
        ("channels last 3d", channels_last_3d, channels_last_3d.stride()),
        (
            "numerical edges",
            module.tensor(memoryview(SPECIAL_INPUT_BITS.view(np.float32))),
            (1,),
        ),
    )


class TensorTruncTests(unittest.TestCase):
    @staticmethod
    def tensor_bits(tensor):
        return np.asarray(tensor, dtype=np.float32).reshape(-1).view(np.uint32)

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

    @staticmethod
    def top_level_calls(source):
        return (
            ("positional", lambda: torch.trunc(source)),
            ("input", lambda: torch.trunc(input=source)),
            ("x", lambda: torch.trunc(x=source)),
            ("a", lambda: torch.trunc(a=source)),
            ("x1", lambda: torch.trunc(x1=source)),
            ("out none", lambda: torch.trunc(source, out=None)),
            ("alias and out none", lambda: torch.trunc(x=source, out=None)),
        )

    def test_values_layouts_offsets_empty_tensors_and_fresh_storage(self):
        for case, source, expected_stride in make_cases(torch):
            output = source.trunc()
            with self.subTest(case=case, metadata=True):
                self.assertEqual(output.shape, source.shape)
                self.assertEqual(output.stride(), expected_stride)
                self.assertEqual(output.storage_offset(), 0)
                self.assertFalse(output.requires_grad)
                self.assertTrue(output.is_leaf)
                self.assertIs(output.dtype, torch.float32)
                self.assertEqual(output.device, torch.device("cpu"))
                self.assertFalse(output.is_set_to(source))
                if source.numel():
                    self.assertNotEqual(output.data_ptr(), source.data_ptr())

            if case == "numerical edges":
                expected_bits = SPECIAL_OUTPUT_BITS
            else:
                values = np.asarray(source, dtype=np.float32).reshape(-1)
                expected_bits = np.trunc(values).astype(np.float32).view(np.uint32)
            with self.subTest(case=case, values=True):
                np.testing.assert_array_equal(
                    self.tensor_bits(output), expected_bits
                )

    def test_top_level_calls_reuse_tensor_trunc_bits_layouts_and_storage(self):
        for case, source, expected_stride in make_cases(torch):
            expected = source.trunc()
            for form, call in self.top_level_calls(source):
                actual = call()
                with self.subTest(case=case, form=form, metadata=True):
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
                with self.subTest(case=case, form=form, values=True):
                    np.testing.assert_array_equal(
                        self.tensor_bits(actual), self.tensor_bits(expected)
                    )

    def test_active_autograd_is_rejected_before_output_planning(self):
        message = r"^trunc\(\): autograd recording is not supported$"
        for case, source in enumerate(self.make_tracked_cases()):
            with self.subTest(case=case):
                with self.assertRaisesRegex(RuntimeError, message):
                    source.trunc()

        extreme = torch.zeros((0,), requires_grad=True).reshape(
            (0, sys.maxsize, 3)
        )
        with self.assertRaisesRegex(RuntimeError, message):
            extreme.trunc()
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
                extreme.trunc()

    def test_detached_and_no_grad_inputs_use_the_inference_path(self):
        for case, source in enumerate(self.make_tracked_cases()):
            detached = source.detach()
            expected = detached.trunc()
            with torch.no_grad():
                actual = source.trunc()
            with self.subTest(case=case, mode="no_grad"):
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.stride(), expected.stride())
                self.assertEqual(actual.storage_offset(), expected.storage_offset())
                self.assertFalse(actual.requires_grad)
                self.assertTrue(actual.is_leaf)
                self.assertFalse(actual.is_set_to(source))
                np.testing.assert_array_equal(
                    self.tensor_bits(actual), self.tensor_bits(expected)
                )
            with self.subTest(case=case, mode="detached"):
                self.assertFalse(expected.is_set_to(detached))
                if detached.numel():
                    self.assertNotEqual(expected.data_ptr(), detached.data_ptr())

    def test_tensorbase_descriptor_metadata_and_no_argument_errors(self):
        tensor = torch.tensor([1.25])
        descriptor = inspect.getattr_static(torch.Tensor, "trunc")
        bound = tensor.trunc

        self.assertIs(torch.Tensor.trunc, descriptor)
        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor), "<method 'trunc' of 'torch._C.TensorBase' objects>"
        )
        self.assertEqual(descriptor.__name__, "trunc")
        self.assertEqual(descriptor.__qualname__, "TensorBase.trunc")
        self.assertEqual(bound.__name__, "trunc")
        self.assertEqual(bound.__qualname__, "Tensor.trunc")
        self.assertEqual(descriptor.__doc__, TRUNC_DOC)
        self.assertEqual(bound.__doc__, TRUNC_DOC)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        assert_no_argument_signature(self, descriptor, "(self, /)")
        assert_no_argument_signature(self, bound, "()")

        cases = (
            (
                lambda: tensor.trunc(1),
                "TensorBase.trunc() takes no arguments (1 given)",
            ),
            (lambda: bound(1), "Tensor.trunc() takes no arguments (1 given)"),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.trunc() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.trunc(1, 2),
                "TensorBase.trunc() takes no arguments (2 given)",
            ),
            (
                lambda: tensor.trunc(input=tensor),
                (
                    "Tensor.trunc() takes no keyword arguments"
                    if sys.version_info < (3, 11)
                    else "TensorBase.trunc() takes no keyword arguments"
                ),
            ),
            (
                lambda: bound(unexpected=True),
                "Tensor.trunc() takes no keyword arguments",
            ),
            (
                lambda: descriptor(tensor, unexpected=True),
                "TensorBase.trunc() takes no keyword arguments",
            ),
            (
                lambda: descriptor(),
                "unbound method TensorBase.trunc() needs an argument",
            ),
            (
                lambda: descriptor(1),
                "descriptor 'trunc' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.trunc() needs an argument",
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
        descriptor = inspect.getattr_static(torch.Tensor, "trunc")
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
            result = tracked.trunc()
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
                forwarded = plain.trunc()
        self.assertEqual(order, ["upper", "lower"])
        self.assertEqual(forwarded.tolist(), [1.0])

        order.clear()
        with self.assertRaisesRegex(
            RuntimeError, r"^trunc\(\): autograd recording is not supported$"
        ):
            with ForwardingMode("lower"):
                with ForwardingMode("upper"):
                    tracked.trunc()
        self.assertEqual(order, ["upper", "lower"])

        old_recursion_limit = sys.getrecursionlimit()
        declining = RecordingMode(NotImplemented)
        try:
            sys.setrecursionlimit(80)
            with declining:
                with self.assertRaises(RecursionError):
                    plain.trunc()
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
                plain.trunc(1)
        self.assertEqual(invalid.calls, [])

    def test_top_level_autograd_rejection_and_no_grad_reuse_method_path(self):
        message = r"^trunc\(\): autograd recording is not supported$"
        for case, source in enumerate(self.make_tracked_cases()):
            detached = source.detach()
            expected = detached.trunc()
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
            torch.trunc(extreme)
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
                torch.trunc(extreme)

    def test_top_level_concrete_out_is_rejected_without_mutation(self):
        source = torch.tensor([1.25, -1.25], requires_grad=True)
        destination = torch.tensor([17.0, 19.0])
        destination_pointer = destination.data_ptr()
        expected_bits = self.tensor_bits(destination).copy()
        for form, call in (
            ("positional", lambda: torch.trunc(source, out=destination)),
            ("keyword", lambda: torch.trunc(input=source, out=destination)),
            ("alias", lambda: torch.trunc(x=source, out=destination)),
        ):
            with self.subTest(form=form):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^trunc\(\): the 'out' argument is not supported$",
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
            self.assertIs(torch.trunc(input=tensor, out=destination), marker)
        self.assertEqual(
            mode.calls,
            [(torch.trunc, (), (), {"input": tensor, "out": destination})],
        )

        override_calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_calls.append((func, types, args, kwargs))
                return marker

        self.assertIs(torch.trunc(Override()), marker)
        self.assertIs(torch.trunc(torch.tensor([1.25]), out=Override()), marker)
        self.assertEqual(len(override_calls), 2)
        for function, dispatch_types, _, _ in override_calls:
            self.assertIs(function, torch.trunc)
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

        self.assertIs(torch.trunc(BaseOverride(), out=DerivedOverride()), marker)
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
                forwarded = torch.trunc(input=plain, out=None)
        self.assertEqual(forwarding_order, ["upper", "lower"])
        self.assertEqual(forwarded.tolist(), [1.0])

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
            self.assertIs(torch.trunc(FallbackOverride()), marker)
        self.assertEqual(events, ["mode", "override"])

        invalid = RecordingMode()
        with self.assertRaises(TypeError):
            with invalid:
                torch.trunc(tensor, tensor)
        self.assertEqual(invalid.calls, [])

    def test_top_level_builtin_metadata_exports_copying_and_pickling(self):
        function = torch.trunc
        self.assertIs(function, torch._C.trunc)
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "trunc")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.trunc")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, TOP_LEVEL_TRUNC_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function), r"^<built-in method trunc of type object at 0x[0-9a-f]+>$"
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.trunc, function)
        for action in (
            lambda: setattr(owner, "trunc", None),
            lambda: delattr(owner, "trunc"),
        ):
            with self.assertRaises(TypeError):
                action()
            self.assertIs(owner.trunc, function)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("trunc"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["trunc"], function)

    def test_top_level_binding_and_type_error_precedence(self):
        tensor = torch.tensor([1.25])
        cases = (
            (
                lambda: torch.trunc(),
                'trunc() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.trunc(tensor, tensor),
                "trunc() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.trunc(tensor, input=tensor),
                "trunc() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.trunc(out=tensor),
                'trunc() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.trunc(extra=tensor),
                'trunc() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.trunc(1, extra=True),
                "trunc(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.trunc(input=[]),
                "trunc(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.trunc(tensor, out=[]),
                "trunc(): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.trunc(tensor, extra=True, out=[]),
                "trunc(): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.trunc(tensor, extra=True),
                "trunc() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.trunc(input=tensor, a=tensor),
                "trunc() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.trunc(a=tensor, x=tensor, out=None),
                "trunc() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.trunc(x=tensor, a=tensor, out=None),
                "trunc() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.trunc(np.zeros((2, 3), dtype=np.float32)),
                (
                    "trunc(): argument 'input' (position 1) must be Tensor, "
                    "not numpy.ndarray"
                ),
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

    def test_inplace_forms_remain_unsupported(self):
        tensor = torch.tensor([1.25])
        self.assertTrue(hasattr(torch, "trunc"))
        self.assertFalse(hasattr(torch.Tensor, "trunc_"))
        self.assertFalse(hasattr(tensor, "trunc_"))
        self.assertFalse(hasattr(torch, "trunc_"))
        self.assertNotIn("trunc_", torch.__all__)
        with self.assertRaises(TypeError):
            tensor.trunc(out=None)


if __name__ == "__main__":
    unittest.main()
