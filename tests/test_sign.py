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


SIGN_DOC = """
sign() -> Tensor

See :func:`torch.sign`
"""

TOP_LEVEL_SIGN_DOC = """
sign(input, *, out=None) -> Tensor

Returns a new tensor with the signs of the elements of :attr:`input`.

.. math::
    \\text{out}_{i} = \\operatorname{sgn}(\\text{input}_{i})

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.tensor([0.7, -1.2, 0., 2.3])
    >>> a
    tensor([ 0.7000, -1.2000,  0.0000,  2.3000])
    >>> torch.sign(a)
    tensor([ 1., -1.,  0.,  1.])
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
        0x3F80_0000,
        0xBF80_0000,
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
        0x0000_0000,
        0x3F80_0000,
        0xBF80_0000,
        0x3F80_0000,
        0xBF80_0000,
        0x3F80_0000,
        0xBF80_0000,
        0x3F80_0000,
        0xBF80_0000,
        0x3F80_0000,
        0xBF80_0000,
        0x3F80_0000,
        0xBF80_0000,
        0x0000_0000,
        0x0000_0000,
        0x0000_0000,
        0x0000_0000,
    ),
    dtype=np.uint32,
)


def make_cases(module=torch):
    base = module.tensor(
        np.linspace(-3.75, 3.75, 24, dtype=np.float32)
        .reshape(2, 3, 4)
        .tolist(),
        dtype=module.float32,
    )
    strided = base.transpose(0, 2)
    return (
        ("scalar", module.tensor(-0.0, dtype=module.float32), ()),
        (
            "empty offset",
            module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
            (2, 1),
        ),
        ("empty singleton trailing", module.zeros((0, 1)), (1, 1)),
        ("contiguous", base, (12, 4, 1)),
        ("offset", strided[1], (1, 3)),
        ("noncontiguous", strided, (1, 4, 12)),
        (
            "numerical edges",
            module.tensor(memoryview(SPECIAL_INPUT_BITS.view(np.float32))),
            (1,),
        ),
    )


def expected_sign_bits(source):
    bits = np.asarray(source, dtype=np.float32).reshape(-1).view(np.uint32)
    magnitude = bits & np.uint32(0x7FFF_FFFF)
    output = np.zeros(bits.shape, dtype=np.uint32)
    positive = (magnitude != 0) & (magnitude <= np.uint32(0x7F80_0000))
    output[positive & ((bits & np.uint32(0x8000_0000)) == 0)] = np.uint32(0x3F80_0000)
    output[positive & ((bits & np.uint32(0x8000_0000)) != 0)] = np.uint32(0xBF80_0000)
    return output


class TensorSignTests(unittest.TestCase):
    @staticmethod
    def tensor_bits(tensor):
        return np.asarray(tensor, dtype=np.float32).reshape(-1).view(np.uint32)

    @staticmethod
    def top_level_calls(source):
        return (
            ("positional", lambda: torch.sign(source)),
            ("input", lambda: torch.sign(input=source)),
            ("x", lambda: torch.sign(x=source)),
            ("a", lambda: torch.sign(a=source)),
            ("x1", lambda: torch.sign(x1=source)),
            ("out none", lambda: torch.sign(source, out=None)),
            ("alias and out none", lambda: torch.sign(x=source, out=None)),
        )

    @staticmethod
    def make_tracked_cases():
        scalar = torch.tensor(-1.25, requires_grad=True)
        empty_leaf = torch.zeros((2, 0, 3), requires_grad=True)
        empty = empty_leaf.transpose(0, 2)[1]
        leaf = torch.tensor(
            np.linspace(-3.75, 3.75, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist(),
            requires_grad=True,
        )
        strided = leaf.transpose(0, 2)
        return (
            (scalar, scalar),
            (empty_leaf, empty),
            (leaf, strided[1]),
            (leaf, strided),
        )

    def assert_result(self, output, source, expected_stride, *, case):
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
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                self.tensor_bits(output), expected_sign_bits(source)
            )

    def test_values_layouts_offsets_empty_tensors_and_fresh_storage(self):
        for case, source, expected_stride in make_cases():
            method_output = source.sign()
            self.assert_result(
                method_output, source, expected_stride, case=(case, "method")
            )
            if case == "numerical edges":
                np.testing.assert_array_equal(
                    self.tensor_bits(method_output), SPECIAL_OUTPUT_BITS
                )

            for form, call in self.top_level_calls(source):
                output = call()
                self.assert_result(
                    output, source, expected_stride, case=(case, form)
                )
                np.testing.assert_array_equal(
                    self.tensor_bits(output), self.tensor_bits(method_output)
                )

    def test_active_autograd_records_reusable_sign_backward_zero_vjp(self):
        for case, (leaf, source) in enumerate(self.make_tracked_cases()):
            with self.subTest(case=case):
                expected = source.detach().sign()
                output = source.sign()
                self.assertEqual(output.shape, expected.shape)
                self.assertEqual(output.stride(), expected.stride())
                self.assertEqual(output.storage_offset(), 0)
                self.assertTrue(output.requires_grad)
                self.assertFalse(output.is_leaf)
                self.assertFalse(output.is_set_to(source))
                np.testing.assert_array_equal(
                    self.tensor_bits(output), self.tensor_bits(expected)
                )
                self.assertEqual(
                    torch._C._nn_functional_dropout_tensor_autograd_suffix(output),
                    ", grad_fn=<SignBackward0>",
                )

                loss = output if output.numel() == 1 else output.sum()
                loss.backward()
                loss.backward()
                gradient_bits = self.tensor_bits(leaf.grad)
                np.testing.assert_array_equal(
                    gradient_bits, np.zeros(gradient_bits.shape, dtype=np.uint32)
                )

    def test_zero_vjp_ignores_upstream_values_and_composes(self):
        leaf = torch.tensor([-1.25, -0.0, 1.75, float("nan")], requires_grad=True)
        weights = torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0])
        (leaf.sign() * weights).sum().backward()
        np.testing.assert_array_equal(
            self.tensor_bits(leaf.grad), np.zeros((4,), dtype=np.uint32)
        )

        composed = torch.tensor([-0.5, 0.5], requires_grad=True)
        composed.sin().sign().sum().backward()
        np.testing.assert_array_equal(
            self.tensor_bits(composed.grad), np.zeros((2,), dtype=np.uint32)
        )

    def test_detached_and_no_grad_inputs_use_the_inference_path(self):
        for case, (_, source) in enumerate(self.make_tracked_cases()):
            detached = source.detach()
            expected = detached.sign()
            with torch.no_grad():
                actual = source.sign()
            with self.subTest(case=case, mode="no_grad"):
                self.assert_result(
                    actual, source, expected.stride(), case=(case, "no_grad")
                )
            with self.subTest(case=case, mode="detached"):
                self.assert_result(
                    expected, detached, expected.stride(), case=(case, "detached")
                )

    def test_tensorbase_descriptor_metadata_and_no_argument_errors(self):
        tensor = torch.tensor([1.25])
        descriptor = inspect.getattr_static(torch.Tensor, "sign")
        bound = tensor.sign

        self.assertIs(torch.Tensor.sign, descriptor)
        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor), "<method 'sign' of 'torch._C.TensorBase' objects>"
        )
        self.assertEqual(descriptor.__name__, "sign")
        self.assertEqual(descriptor.__qualname__, "TensorBase.sign")
        self.assertEqual(bound.__name__, "sign")
        self.assertEqual(bound.__qualname__, "Tensor.sign")
        self.assertEqual(descriptor.__doc__, SIGN_DOC)
        self.assertEqual(bound.__doc__, SIGN_DOC)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        assert_no_argument_signature(self, descriptor, "(self, /)")
        assert_no_argument_signature(self, bound, "()")

        cases = (
            (lambda: tensor.sign(1), "TensorBase.sign() takes no arguments (1 given)"),
            (lambda: bound(1), "Tensor.sign() takes no arguments (1 given)"),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.sign() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.sign(1, 2),
                "TensorBase.sign() takes no arguments (2 given)",
            ),
            (
                lambda: tensor.sign(input=tensor),
                (
                    "Tensor.sign() takes no keyword arguments"
                    if sys.version_info < (3, 11)
                    else "TensorBase.sign() takes no keyword arguments"
                ),
            ),
            (lambda: bound(unexpected=True), "Tensor.sign() takes no keyword arguments"),
            (
                lambda: descriptor(tensor, unexpected=True),
                "TensorBase.sign() takes no keyword arguments",
            ),
            (lambda: descriptor(), "unbound method TensorBase.sign() needs an argument"),
            (
                lambda: descriptor(1),
                "descriptor 'sign' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.sign() needs an argument",
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
        descriptor = inspect.getattr_static(torch.Tensor, "sign")
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
            result = tracked.sign()
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
                forwarded = plain.sign()
        self.assertEqual(order, ["upper", "lower"])
        self.assertEqual(forwarded.tolist(), [1.0])

        order.clear()
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                tracked_forwarded = tracked.sign()
        self.assertEqual(order, ["upper", "lower"])
        self.assertTrue(tracked_forwarded.requires_grad)
        self.assertFalse(tracked_forwarded.is_leaf)
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(
                tracked_forwarded
            ),
            ", grad_fn=<SignBackward0>",
        )

        invalid = RecordingMode(marker)
        with self.assertRaises(TypeError):
            with invalid:
                plain.sign(1)
        self.assertEqual(invalid.calls, [])

    def test_top_level_autograd_and_no_grad_reuse_method_path(self):
        for case, (_, source) in enumerate(self.make_tracked_cases()):
            expected = source.detach().sign()
            for form, call in self.top_level_calls(source):
                with self.subTest(case=case, form=form, mode="recording"):
                    actual = call()
                    self.assertEqual(actual.shape, expected.shape)
                    self.assertEqual(actual.stride(), expected.stride())
                    self.assertEqual(actual.storage_offset(), 0)
                    self.assertTrue(actual.requires_grad)
                    self.assertFalse(actual.is_leaf)
                    self.assertFalse(actual.is_set_to(source))
                    np.testing.assert_array_equal(
                        self.tensor_bits(actual), self.tensor_bits(expected)
                    )
                    self.assertEqual(
                        torch._C._nn_functional_dropout_tensor_autograd_suffix(
                            actual
                        ),
                        ", grad_fn=<SignBackward0>",
                    )
                with self.subTest(case=case, form=form, mode="no_grad"):
                    with torch.no_grad():
                        actual = call()
                    self.assertEqual(actual.shape, expected.shape)
                    self.assertEqual(actual.stride(), expected.stride())
                    self.assertEqual(actual.storage_offset(), 0)
                    self.assertFalse(actual.requires_grad)
                    self.assertTrue(actual.is_leaf)
                    self.assertFalse(actual.is_set_to(source))
                    np.testing.assert_array_equal(
                        self.tensor_bits(actual), self.tensor_bits(expected)
                    )

    def test_top_level_concrete_out_is_rejected_without_mutation(self):
        source = torch.tensor([1.25, -0.0, -2.0], requires_grad=True)
        destination = torch.tensor([17.0, 19.0, 23.0])
        destination_bits = self.tensor_bits(destination).copy()
        destination_pointer = destination.data_ptr()
        for form, call in (
            ("positional", lambda: torch.sign(source, out=destination)),
            ("keyword", lambda: torch.sign(input=source, out=destination)),
            ("alias", lambda: torch.sign(x=source, out=destination)),
        ):
            with self.subTest(form=form):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^sign\(\): the 'out' argument is not supported$",
                ):
                    call()
                self.assertEqual(destination.data_ptr(), destination_pointer)
                np.testing.assert_array_equal(
                    self.tensor_bits(destination), destination_bits
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
            self.assertIs(torch.sign(input=tensor, out=destination), marker)
        self.assertEqual(
            mode.calls,
            [(torch.sign, (), (), {"input": tensor, "out": destination})],
        )

        override_calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_calls.append((func, types, args, kwargs))
                return marker

        self.assertIs(torch.sign(Override()), marker)
        self.assertIs(torch.sign(torch.tensor([1.25]), out=Override()), marker)
        self.assertEqual(len(override_calls), 2)
        for function, dispatch_types, _, _ in override_calls:
            self.assertIs(function, torch.sign)
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

        self.assertIs(torch.sign(BaseOverride(), out=DerivedOverride()), marker)
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
                forwarded = torch.sign(input=plain, out=None)
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
            self.assertIs(torch.sign(FallbackOverride()), marker)
        self.assertEqual(events, ["mode", "override"])

        declining_override_error = r"^Multiple dispatch failed for 'torch.sign'"

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaisesRegex(TypeError, declining_override_error):
            torch.sign(DecliningOverride())

    def test_callable_metadata_documentation_pickling_and_exports(self):
        function = torch.sign
        self.assertIs(function, torch._C.sign)
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "sign")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.sign")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, TOP_LEVEL_SIGN_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function), r"^<built-in method sign of type object at 0x[0-9a-f]+>$"
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.sign, function)
        for action in (
            lambda: setattr(owner, "sign", None),
            lambda: delattr(owner, "sign"),
        ):
            with self.assertRaises(TypeError):
                action()
            self.assertIs(owner.sign, function)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

        self.assertEqual(torch.__all__.count("sign"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["sign"], function)

    def test_binding_type_and_unsupported_extension_errors(self):
        tensor = torch.tensor([1.25])
        cases = (
            (
                lambda: torch.sign(),
                'sign() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.sign(tensor, tensor),
                "sign() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.sign(tensor, input=tensor),
                "sign() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.sign(out=tensor),
                'sign() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.sign(extra=tensor),
                'sign() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.sign(1, extra=True),
                "sign(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.sign(input=[]),
                "sign(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.sign(tensor, out=[]),
                "sign(): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.sign(tensor, extra=True, out=[]),
                "sign(): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.sign(tensor, extra=True),
                "sign() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.sign(input=tensor, a=tensor),
                "sign() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.sign(a=tensor, x=tensor, out=None),
                "sign() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.sign(x=tensor, a=tensor, out=None),
                "sign() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.sign(np.zeros((2, 3), dtype=np.float32)),
                "sign(): argument 'input' (position 1) must be Tensor, not numpy.ndarray",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

        self.assertFalse(hasattr(torch.Tensor, "sign_"))
        self.assertFalse(hasattr(tensor, "sign_"))
        self.assertFalse(hasattr(torch, "sign_"))
        self.assertNotIn("sign_", torch.__all__)
        self.assertFalse(hasattr(torch.Tensor, "sgn"))
        self.assertFalse(hasattr(tensor, "sgn"))
        self.assertFalse(hasattr(torch, "sgn"))
        self.assertNotIn("sgn", torch.__all__)
        self.assertFalse(hasattr(torch, "float64"))
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tensor\(\): device 'cuda' is not supported; only 'cpu' is implemented$",
        ):
            torch.tensor([1.0], device="cuda")

        try:
            tensor_subclass = type("TensorSubclass", (torch.Tensor,), {})
        except TypeError:
            return
        with self.assertRaises((TypeError, RuntimeError)):
            torch.sign(tensor_subclass.__new__(tensor_subclass))


if __name__ == "__main__":
    unittest.main()
