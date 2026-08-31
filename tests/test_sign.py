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
        0x3EAA_AAAB,
        0xBEAA_AAAB,
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
        0x3F80_0000,
        0xBF80_0000,
        0x0000_0000,
        0x0000_0000,
        0x0000_0000,
        0x0000_0000,
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
    return (
        ("scalar", module.tensor(-0.0, dtype=module.float32), ()),
        (
            "empty",
            module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
            (2, 1),
        ),
        ("contiguous", base, base.stride()),
        ("offset", strided[1], (1, 3)),
        ("noncontiguous", strided, (1, 4, 12)),
        (
            "numerical edges",
            module.tensor(memoryview(SPECIAL_INPUT_BITS.view(np.float32))),
            (1,),
        ),
    )


class TensorSignTests(unittest.TestCase):
    @staticmethod
    def tensor_bits(tensor):
        return np.asarray(tensor, dtype=np.float32).reshape(-1).view(np.uint32)

    @staticmethod
    def expected_sign_bits(source):
        source_bits = TensorSignTests.tensor_bits(source)
        magnitude = source_bits & np.uint32(0x7FFF_FFFF)
        expected = np.zeros(source_bits.shape, dtype=np.uint32)
        finite_nonzero = (magnitude != 0) & (magnitude <= np.uint32(0x7F80_0000))
        expected[finite_nonzero & ((source_bits & np.uint32(0x8000_0000)) == 0)] = np.uint32(
            0x3F80_0000
        )
        expected[finite_nonzero & ((source_bits & np.uint32(0x8000_0000)) != 0)] = np.uint32(
            0xBF80_0000
        )
        return expected

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
                self.tensor_bits(output), self.expected_sign_bits(source)
            )

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
        scalar = torch.tensor(-0.0, requires_grad=True)
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
            ("scalar", scalar, scalar),
            ("empty", empty_leaf, empty),
            ("contiguous", leaf, leaf),
            ("offset", leaf, strided[1]),
            ("noncontiguous", leaf, strided),
        )

    def test_values_layouts_offsets_empty_tensors_and_fresh_storage(self):
        for case, source, expected_stride in make_cases(torch):
            output = source.sign()
            self.assert_result(output, source, expected_stride, case=(case, "method"))

            expected = source.sign()
            for form, call in self.top_level_calls(source):
                actual = call()
                self.assert_result(actual, source, expected_stride, case=(case, form))
                np.testing.assert_array_equal(
                    self.tensor_bits(actual), self.tensor_bits(expected)
                )

    def test_active_autograd_records_reusable_sign_backward_zero_vjp(self):
        for case, leaf, source in self.make_tracked_cases():
            for form, call in (("method", source.sign), *self.top_level_calls(source)):
                with self.subTest(case=case, form=form):
                    expected = source.detach().sign()
                    output = call()
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

    def test_zero_vjp_ignores_upstream_values_composes_and_accumulates(self):
        leaf = torch.tensor([-1.25, -0.0, 1.75, 4.5], requires_grad=True)
        weights = torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0])
        (leaf.sign() * weights).sum().backward()
        np.testing.assert_array_equal(
            self.tensor_bits(leaf.grad), np.zeros((4,), dtype=np.uint32)
        )

        accumulated = torch.tensor([-2.0, 0.0, 3.0], requires_grad=True)
        (accumulated * 3.0).sum().backward()
        first = self.tensor_bits(accumulated.grad).copy()
        accumulated.sign().sum().backward()
        np.testing.assert_array_equal(self.tensor_bits(accumulated.grad), first)

        composed = torch.tensor([-0.5, 0.5], requires_grad=True)
        loss = composed.sin().sign().sum()
        loss.backward()
        np.testing.assert_array_equal(
            self.tensor_bits(composed.grad), np.zeros((2,), dtype=np.uint32)
        )
        with self.assertRaisesRegex(
            RuntimeError, "backward through the graph a second time"
        ):
            loss.backward()

    def test_detached_and_no_grad_inputs_use_the_inference_path(self):
        for case, _, source in self.make_tracked_cases():
            detached = source.detach()
            expected = detached.sign()
            with torch.no_grad():
                actual = source.sign()
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
                lambda: tensor.sign(input=tensor),
                (
                    "Tensor.sign() takes no keyword arguments"
                    if sys.version_info < (3, 11)
                    else "TensorBase.sign() takes no keyword arguments"
                ),
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_torch_function_modes_dispatch_before_native_execution(self):
        tracked = torch.tensor([1.25], requires_grad=True)
        plain = tracked.detach()
        descriptor = inspect.getattr_static(torch.Tensor, "sign")
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
            result = tracked.sign()
        self.assertIs(result, marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, (torch.Tensor,))
        self.assertEqual(len(args), 1)
        self.assertIs(args[0], tracked)
        self.assertIsNone(kwargs)

        forwarding_order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                forwarding_order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = plain.sign()
        self.assertEqual(forwarding_order, ["upper", "lower"])
        self.assertEqual(forwarded.tolist(), [1.0])

        top_level_mode = RecordingMode()
        destination = torch.tensor([0.0])
        with top_level_mode:
            self.assertIs(torch.sign(input=tracked, out=destination), marker)
        self.assertEqual(
            top_level_mode.calls,
            [(torch.sign, (), (), {"input": tracked, "out": destination})],
        )

        override_calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_calls.append((func, types, args, kwargs))
                return marker

        self.assertIs(torch.sign(Override()), marker)
        self.assertIs(torch.sign(plain, out=Override()), marker)
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

    def test_top_level_concrete_out_is_rejected_without_mutation(self):
        source = torch.tensor([1.25, -1.25], requires_grad=True)
        destination = torch.tensor([17.0, 19.0])
        destination_pointer = destination.data_ptr()
        expected_bits = self.tensor_bits(destination).copy()
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
                    self.tensor_bits(destination), expected_bits
                )
                self.assertIsNone(source.grad)

    def test_top_level_builtin_metadata_exports_copying_and_pickling(self):
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
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("sign"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["sign"], function)

    def test_top_level_binding_type_and_unsupported_extension_errors(self):
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
                (
                    "sign(): argument 'input' (position 1) must be Tensor, "
                    "not numpy.ndarray"
                ),
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

    def test_sgn_and_inplace_forms_remain_unsupported(self):
        tensor = torch.tensor([1.25])
        self.assertTrue(hasattr(torch, "sign"))
        self.assertTrue(hasattr(torch.Tensor, "sign"))
        self.assertFalse(hasattr(torch, "sgn"))
        self.assertFalse(hasattr(torch.Tensor, "sgn"))
        self.assertFalse(hasattr(tensor, "sgn"))
        self.assertFalse(hasattr(torch.Tensor, "sign_"))
        self.assertFalse(hasattr(tensor, "sign_"))
        self.assertFalse(hasattr(torch, "sign_"))
        self.assertNotIn("sgn", torch.__all__)
        self.assertNotIn("sign_", torch.__all__)
        with self.assertRaises(TypeError):
            tensor.sign(out=None)


if __name__ == "__main__":
    unittest.main()
