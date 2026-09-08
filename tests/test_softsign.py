import copy
import inspect
import pickle
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch
import torch_rs.nn as nn
import torch_rs.nn.functional as functional

if __package__:
    from .signature_utils import assert_no_argument_signature
else:
    from signature_utils import assert_no_argument_signature


TENSOR_DOC = """
softsign() -> Tensor

See :func:`torch.nn.functional.softsign`
"""

TOP_LEVEL_DOC = r"""
softsign(input, *, out=None) -> Tensor

Applies element-wise, the function :math:`\text{SoftSign}(x) = \frac{x}{1 + |x|}`

See :class:`~torch.nn.Softsign` for more details.
"""


class SoftsignAliasTests(unittest.TestCase):
    @staticmethod
    def tensor_bits(tensor):
        return np.asarray(tensor, dtype=np.float32).reshape(-1).view(np.uint32)

    @classmethod
    def tensor_state(cls, tensor):
        return (
            tensor.shape,
            tensor.stride(),
            tensor.storage_offset(),
            tensor.data_ptr(),
            cls.tensor_bits(tensor).copy(),
        )

    @staticmethod
    def make_cases():
        base = torch.tensor(
            np.linspace(-3.0, 3.0, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist()
        )
        strided = base.transpose(0, 2)
        rank_two = torch.tensor(
            [[-4.0, -0.0, 0.5], [1.0, 2.0, 8.0]],
        )
        return (
            ("scalar", torch.tensor(-0.0)),
            ("empty", torch.zeros((2, 0, 3)).transpose(0, 2)[1]),
            ("contiguous", base),
            ("offset", base[1]),
            ("noncontiguous", strided[1]),
            ("rank2", rank_two),
        )

    @staticmethod
    def top_level_calls(source):
        return (
            ("positional", lambda: torch.softsign(source)),
            ("input", lambda: torch.softsign(input=source)),
            ("x", lambda: torch.softsign(x=source)),
            ("a", lambda: torch.softsign(a=source)),
            ("x1", lambda: torch.softsign(x1=source)),
            ("out none", lambda: torch.softsign(source, out=None)),
            ("alias and out none", lambda: torch.softsign(x=source, out=None)),
        )

    def assert_matches_functional(self, actual, expected, source, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
            self.assertFalse(actual.is_set_to(source))
            self.assertFalse(actual.is_set_to(expected))
            if source.numel():
                self.assertNotEqual(actual.data_ptr(), source.data_ptr())
        np.testing.assert_array_equal(
            self.tensor_bits(actual),
            self.tensor_bits(expected),
        )

    def test_tensor_method_matches_functional_softsign(self):
        for case, source in self.make_cases():
            before = self.tensor_state(source)
            expected = functional.softsign(source)
            actual = source.softsign()
            self.assert_matches_functional(actual, expected, source, case=case)
            after = self.tensor_state(source)
            with self.subTest(case=case, nonmutation=True):
                self.assertEqual(after[:-1], before[:-1])
                np.testing.assert_array_equal(after[-1], before[-1])

    def test_top_level_calls_match_functional_softsign(self):
        for case, source in self.make_cases():
            expected = functional.softsign(source)
            for form, call in self.top_level_calls(source):
                actual = call()
                self.assert_matches_functional(
                    actual,
                    expected,
                    source,
                    case=(case, form),
                )

    def test_numerical_edges_match_functional_softsign_bitwise(self):
        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x0080_0000,
                0x8080_0000,
                0x3EAA_AAAB,
                0xBEAA_AAAB,
                0x3F80_0000,
                0xBF80_0000,
                0x4000_0000,
                0xC000_0000,
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
        source = torch.tensor(memoryview(special_bits.view(np.float32)))
        expected = functional.softsign(source)
        np.testing.assert_array_equal(
            self.tensor_bits(source.softsign()),
            self.tensor_bits(expected),
        )
        np.testing.assert_array_equal(
            self.tensor_bits(torch.softsign(source)),
            self.tensor_bits(expected),
        )

    def test_top_level_builtin_metadata_exports_copying_and_pickling(self):
        function = torch.softsign
        self.assertIs(function, torch._C.softsign)
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "softsign")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.softsign")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, TOP_LEVEL_DOC)
        self.assertIsNone(function.__text_signature__)
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.softsign, function)
        for action in (
            lambda: setattr(owner, "softsign", None),
            lambda: delattr(owner, "softsign"),
        ):
            with self.assertRaises(TypeError):
                action()
            self.assertIs(owner.softsign, function)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("softsign"), 1)
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["softsign"], function)

    def test_tensor_method_metadata_copy_and_pickle(self):
        descriptor = inspect.getattr_static(torch.Tensor, "softsign")
        bound = torch.tensor([0.5]).softsign
        self.assertIs(torch.Tensor.softsign, descriptor)
        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor), "<method 'softsign' of 'torch._C.TensorBase' objects>"
        )
        self.assertEqual(descriptor.__name__, "softsign")
        self.assertEqual(descriptor.__qualname__, "TensorBase.softsign")
        self.assertEqual(bound.__name__, "softsign")
        self.assertEqual(bound.__qualname__, "Tensor.softsign")
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertEqual(descriptor.__doc__, TENSOR_DOC)
        self.assertEqual(bound.__doc__, TENSOR_DOC)
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        assert_no_argument_signature(self, descriptor, "(self, /)")
        assert_no_argument_signature(self, bound, "()")
        self.assertIs(copy.copy(descriptor), descriptor)
        self.assertIs(copy.deepcopy(descriptor), descriptor)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(descriptor, protocol=protocol)),
                    descriptor,
                )

    def test_active_autograd_is_rejected_but_no_grad_and_detached_inputs_work(self):
        leaf = torch.tensor(
            [[-2.0, -0.0, 1.0], [2.0, 4.0, 8.0]], requires_grad=True
        )
        for case, source in (
            ("method scalar", torch.tensor(0.5, requires_grad=True)),
            ("method noncontiguous", leaf.transpose(0, 1)[1]),
            ("top level scalar", torch.tensor(0.5, requires_grad=True)),
            ("top level noncontiguous", leaf.transpose(0, 1)[1]),
        ):
            before = self.tensor_state(source)
            call = source.softsign if case.startswith("method") else (
                lambda source=source: torch.softsign(source)
            )
            with self.subTest(case=case):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^softsign\(\): autograd recording is not supported$",
                ):
                    call()
                self.assertIsNone(source.grad)
                after = self.tensor_state(source)
                self.assertEqual(after[:-1], before[:-1])
                np.testing.assert_array_equal(after[-1], before[-1])

        source = leaf.transpose(0, 1)[1]
        with torch.no_grad():
            expected = functional.softsign(source)
            self.assert_matches_functional(
                source.softsign(),
                expected,
                source,
                case="method no_grad",
            )
            self.assert_matches_functional(
                torch.softsign(source),
                expected,
                source,
                case="top level no_grad",
            )

        detached = source.detach()
        expected = functional.softsign(detached)
        self.assert_matches_functional(
            detached.softsign(),
            expected,
            detached,
            case="method detached",
        )
        self.assert_matches_functional(
            torch.softsign(detached),
            expected,
            detached,
            case="top level detached",
        )
        self.assertIsNone(leaf.grad)

    def test_argument_errors_and_unsupported_boundaries(self):
        tensor = torch.tensor([0.5])
        destination = torch.tensor([17.0])
        cases = (
            (
                lambda: torch.softsign(),
                TypeError,
                'softsign() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.softsign(tensor, tensor),
                TypeError,
                "softsign() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.softsign(tensor, input=tensor),
                TypeError,
                "softsign() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.softsign(out=tensor),
                TypeError,
                'softsign() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.softsign(1),
                TypeError,
                "softsign(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.softsign(input=[]),
                TypeError,
                "softsign(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.softsign(tensor, out=[]),
                TypeError,
                "softsign(): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.softsign(tensor, extra=True),
                TypeError,
                "softsign() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: tensor.softsign(out=None),
                TypeError,
                None,
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(error_type) as raised:
                    call()
                if message is not None:
                    self.assertEqual(str(raised.exception), message)

        with self.assertRaisesRegex(
            RuntimeError,
            r"^softsign\(\): the 'out' argument is not supported$",
        ):
            torch.softsign(tensor, out=destination)
        self.assertEqual(destination.tolist(), [17.0])

        with self.assertRaisesRegex(
            TypeError, r"^type 'torch_rs\.Tensor' is not an acceptable base type$"
        ):
            type("TensorSubclass", (torch.Tensor,), {})
        self.assertFalse(hasattr(torch, "softsign_"))
        self.assertFalse(hasattr(torch.Tensor, "softsign_"))
        self.assertFalse(hasattr(nn, "Softsign"))
        self.assertFalse(hasattr(functional, "softsign_"))
        self.assertFalse(hasattr(torch, "float64"))
        with self.assertRaisesRegex(
            TypeError,
            r"^tensor\(\): argument 'dtype' must be torch\.dtype, not object$",
        ):
            torch.tensor([1.0], dtype=object()).softsign()
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tensor\(\): device 'cuda' is not supported; only 'cpu' is implemented$",
        ):
            torch.tensor([1.0], device="cuda").softsign()

    def test_torch_function_overrides_and_modes_are_rejected(self):
        marker = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        with self.assertRaisesRegex(
            TypeError,
            r"^softsign\(\): argument 'input' \(position 1\) must be Tensor, not Override$",
        ):
            torch.softsign(Override())
        self.assertEqual(Override.calls, [])

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        tensor = torch.tensor([0.5])
        for case, call in (
            ("top level", lambda: torch.softsign(tensor)),
            ("method", tensor.softsign),
        ):
            mode = RecordingMode()
            with self.subTest(case=case):
                with mode:
                    with self.assertRaisesRegex(
                        TypeError,
                        r"^softsign\(\) does not support an active TorchFunctionMode$",
                    ):
                        call()
                self.assertEqual(mode.calls, [])
                self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])


if __name__ == "__main__":
    unittest.main()
