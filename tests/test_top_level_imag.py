import copy
import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


ERROR = "imag is not implemented for tensors with non-complex dtypes."
FUNCTION_DOC = (
    "\nimag(input) -> Tensor\n\n"
    "Returns a new tensor containing imaginary values of the :attr:`self` tensor.\n"
    "The returned tensor and :attr:`self` share the same underlying storage.\n\n"
    ".. warning::\n"
    "    :func:`imag` is only supported for tensors with complex dtypes.\n\n"
    "Args:\n"
    "    input (Tensor): the input tensor.\n\n"
    "Example::\n\n"
    "    >>> x=torch.randn(4, dtype=torch.cfloat)\n"
    "    >>> x\n"
    "    tensor([(0.3100+0.3553j), (-0.5445-0.7896j), "
    "(-1.6492-0.0633j), (-0.0638-0.8119j)])\n"
    "    >>> x.imag\n"
    "    tensor([ 0.3553, -0.7896, -0.0633, -0.8119])\n\n"
)


class TopLevelImagTests(unittest.TestCase):
    def assert_imag_error(self, action):
        with self.assertRaises(RuntimeError) as raised:
            action()
        self.assertIs(type(raised.exception), RuntimeError)
        self.assertEqual(str(raised.exception), ERROR)
        self.assertEqual(raised.exception.args, (ERROR,))
        return raised.exception

    @staticmethod
    def metadata(tensor):
        return (
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

    @staticmethod
    def calls(tensor):
        return (
            ("positional", lambda: torch.imag(tensor)),
            ("input", lambda: torch.imag(input=tensor)),
            ("x", lambda: torch.imag(x=tensor)),
            ("a", lambda: torch.imag(a=tensor)),
            ("x1", lambda: torch.imag(x1=tensor)),
        )

    def tensor_cases(self):
        scalar_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x007F_FFFF,
                0x0080_0000,
                0x3F80_0000,
                0x7F7F_FFFF,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        scalar_storage = torch.tensor(memoryview(scalar_bits.view(np.float32)))
        base = torch.tensor(
            np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5).tolist()
        )
        strided = base.transpose(0, 3)
        empty = torch.zeros((2, 0, 3)).transpose(0, 2)[1]
        leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        non_leaf = (leaf * 3.0).transpose(0, 1)[1]
        with torch.no_grad():
            no_grad_view = leaf.transpose(0, 1)

        self.assertGreater(empty.storage_offset(), 0)
        self.assertGreater(strided[1].storage_offset(), 0)
        self.assertFalse(strided.is_contiguous())
        return leaf, non_leaf, (
            *(
                (f"float32 bits 0x{bits:08x}", scalar_storage[index])
                for index, bits in enumerate(scalar_bits)
            ),
            ("contiguous", base),
            ("empty offset view", empty),
            ("offset strided view", strided[1]),
            ("noncontiguous view", strided),
            ("autograd leaf", leaf),
            ("autograd non-leaf", non_leaf),
            ("detached non-leaf", non_leaf.detach()),
            ("no-grad leaf view", no_grad_view),
        )

    def test_all_supported_call_forms_raise_without_side_effects(self):
        leaf, non_leaf, cases = self.tensor_cases()
        for case, tensor in cases:
            for form, call in self.calls(tensor):
                with self.subTest(case=case, form=form):
                    metadata = self.metadata(tensor)
                    alias = tensor.detach()
                    bits = np.asarray(alias).reshape(-1).view(np.uint32).copy()

                    errors = [self.assert_imag_error(call) for _ in range(3)]

                    self.assertEqual(len({id(error) for error in errors}), len(errors))
                    self.assertEqual(self.metadata(tensor), metadata)
                    self.assertTrue(tensor.is_set_to(alias))
                    np.testing.assert_array_equal(
                        np.asarray(tensor.detach()).reshape(-1).view(np.uint32),
                        bits,
                    )

        self.assertIsNone(leaf.grad)
        non_leaf.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[0.0, 3.0, 0.0], [0.0, 3.0, 0.0]])
        gradient = leaf.grad
        self.assert_imag_error(lambda: torch.imag(leaf))
        self.assertIs(leaf.grad, gradient)

    def test_modes_and_overrides_observe_the_original_call(self):
        tensor = torch.tensor([1.0], requires_grad=True)
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return self.result

        mode = RecordingMode()
        with mode:
            self.assertIs(torch.imag(input=tensor), marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, torch.imag)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {"input": tensor})

        mode = RecordingMode()
        with mode:
            self.assertIs(torch.imag(tensor), marker)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, torch.imag)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, (tensor,))
        self.assertIsNone(kwargs)

        override_calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                override_calls.append((func, dispatch_types, args, kwargs))
                return marker

        for form, call in (
            ("positional", lambda: torch.imag(Override())),
            ("input", lambda: torch.imag(input=Override())),
            ("x", lambda: torch.imag(x=Override())),
            ("a", lambda: torch.imag(a=Override())),
            ("x1", lambda: torch.imag(x1=Override())),
        ):
            with self.subTest(form=form):
                self.assertIs(call(), marker)
        self.assertEqual(len(override_calls), 5)
        for function, dispatch_types, _, _ in override_calls:
            self.assertIs(function, torch.imag)
            self.assertEqual(dispatch_types, (Override,))

        forwarding_order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                forwarding_order.append(self.label)
                return func(*args, **(kwargs or {}))

        with self.assertRaisesRegex(RuntimeError, f"^{ERROR}$"):
            with ForwardingMode("lower"):
                with ForwardingMode("upper"):
                    torch.imag(input=tensor)
        self.assertEqual(forwarding_order, ["upper", "lower"])

        class DecliningMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = 0

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls += 1
                return NotImplemented

        mode = DecliningMode()
        with self.assertRaisesRegex(
            TypeError,
            r"Multiple dispatch failed for 'torch\.imag'; all __torch_function__ "
            r"handlers returned NotImplemented:",
        ):
            with mode:
                torch.imag(tensor)
        self.assertEqual(mode.calls, 1)

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaisesRegex(
            TypeError,
            r"Multiple dispatch failed for 'torch\.imag'; all __torch_function__ "
            r"handlers returned NotImplemented:",
        ):
            torch.imag(DecliningOverride())

    def test_callable_metadata_copy_pickling_and_exports(self):
        function = torch.imag
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "imag")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.imag")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function),
            r"^<built-in method imag of type object at 0x[0-9a-f]+>$",
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.imag, function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("imag"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["imag"], function)

        message = (
            "cannot set 'imag' attribute of immutable type "
            "'torch_rs._C._VariableFunctionsClass'"
        )
        with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
            owner.imag = None
        self.assertIs(owner.imag, function)

    def test_binding_type_and_unsupported_scope_errors(self):
        tensor = torch.tensor([1.0])
        cases = (
            (lambda: torch.imag(), 'imag() missing 1 required positional arguments: "input"'),
            (
                lambda: torch.imag(tensor, tensor),
                "imag() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.imag(tensor, input=tensor),
                "imag() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.imag(out=tensor),
                'imag() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.imag(1, extra=True),
                "imag(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.imag(input=[]),
                "imag(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.imag(tensor, out=[]),
                "imag() got an unexpected keyword argument 'out'",
            ),
            (
                lambda: torch.imag(tensor, extra=True, out=[]),
                "imag() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.imag(tensor, extra=True),
                "imag() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.imag(input=tensor, a=tensor),
                "imag() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.imag(a=tensor, x=tensor, out=None),
                "imag() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.imag(x=tensor, a=tensor, out=None),
                "imag() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.imag(np.zeros((2, 3), dtype=np.float32)),
                "imag(): argument 'input' (position 1) must be Tensor, not numpy.ndarray",
            ),
            (
                lambda: torch.imag(tensor, out=None),
                "imag() got an unexpected keyword argument 'out'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

        for name in (
            "complex32",
            "complex64",
            "complex128",
            "chalf",
            "cfloat",
            "cdouble",
        ):
            with self.subTest(dtype=name):
                self.assertFalse(hasattr(torch, name))


if __name__ == "__main__":
    unittest.main()
