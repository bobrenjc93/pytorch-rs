import copy
import importlib
import inspect
import pickle
import sys
import types
import unittest

import numpy as np
import torch_rs as torch
import torch_rs.nn as nn
import torch_rs.nn.functional as functional


FUNCTION_DOC = r"""sigmoid(input) -> Tensor

    Applies the element-wise function :math:`\text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}`

    See :class:`~torch.nn.Sigmoid` for more details.
    """

if sys.version_info >= (3, 13):
    FUNCTION_DOC = (
        "sigmoid(input) -> Tensor\n\n"
        "Applies the element-wise function "
        r":math:`\text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}`"
        "\n\n"
        "See :class:`~torch.nn.Sigmoid` for more details.\n"
    )


class FunctionalSigmoidTests(unittest.TestCase):
    @staticmethod
    def tensor_bits(tensor):
        return np.asarray(tensor).reshape(-1).view(np.uint32)

    def assert_tensor_matches(self, actual, expected, source, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, expected.dtype)
            self.assertEqual(actual.device, expected.device)
            self.assertFalse(actual.is_set_to(source))
            self.assertFalse(actual.is_set_to(expected))
            if source.numel():
                self.assertNotEqual(actual.data_ptr(), source.data_ptr())

        np.testing.assert_array_equal(
            self.tensor_bits(actual),
            self.tensor_bits(expected),
        )

    def test_imports_signature_documentation_copy_and_pickle(self):
        imported_nn = importlib.import_module("torch_rs.nn")
        imported_functional = importlib.import_module("torch_rs.nn.functional")
        from torch_rs.nn import functional as from_nn
        from torch_rs.nn.functional import sigmoid

        self.assertIs(torch.nn, nn)
        self.assertIs(nn, imported_nn)
        self.assertIs(nn.functional, functional)
        self.assertIs(functional, imported_functional)
        self.assertIs(from_nn, functional)
        self.assertIs(sigmoid, functional.sigmoid)
        self.assertFalse(hasattr(nn, "__all__"))
        self.assertFalse(hasattr(functional, "__all__"))

        function = functional.sigmoid
        signature = inspect.signature(function)
        parameter = tuple(signature.parameters.values())[0]
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "sigmoid")
        self.assertEqual(function.__qualname__, "sigmoid")
        self.assertEqual(function.__module__, "torch_rs.nn.functional")
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__annotations__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertEqual(str(signature), "(input)")
        self.assertEqual(parameter.kind, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        self.assertIs(parameter.annotation, inspect.Parameter.empty)
        self.assertIs(signature.return_annotation, inspect.Signature.empty)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

    def test_delegates_values_layout_and_storage_to_tensor_sigmoid(self):
        base = torch.tensor(
            np.linspace(-3.0, 3.0, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist()
        )
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
        cases = (
            ("scalar", torch.tensor(-0.0)),
            ("empty", torch.zeros((2, 0, 3)).transpose(0, 2)[1]),
            ("offset", base[1]),
            ("noncontiguous", base.transpose(0, 2)[1]),
            ("channels_last", channels_last),
            ("channels_last_3d", channels_last_3d),
        )

        for case, source in cases:
            expected = source.sigmoid()
            actual = functional.sigmoid(input=source)
            self.assert_tensor_matches(actual, expected, source, case=case)

    def test_direct_receiver_and_subclass_method_semantics(self):
        marker = object()
        calls = []

        class BaseReceiver:
            def sigmoid(self):
                calls.append(("base", self))
                return object()

        class DerivedReceiver(BaseReceiver):
            def sigmoid(self):
                calls.append(("derived", self))
                return marker

        receiver = DerivedReceiver()
        self.assertIs(functional.sigmoid(receiver), marker)
        self.assertEqual(calls, [("derived", receiver)])

        class TorchFunctionReceiver:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                raise AssertionError("functional.sigmoid must delegate to the method")

            def sigmoid(self):
                return marker

        self.assertIs(functional.sigmoid(TorchFunctionReceiver()), marker)

    def test_modes_observe_the_tensorbase_method_descriptor(self):
        source = torch.tensor([0.5], requires_grad=True)
        descriptor = inspect.getattr_static(torch.Tensor, "sigmoid")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = functional.sigmoid(input=source)
            self.assertEqual(
                torch.overrides._get_current_function_mode_stack(), [mode]
            )
        self.assertIs(result, marker)
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, (torch.Tensor,))
        self.assertEqual(args, (source,))
        self.assertIsNone(kwargs)

    def test_scalar_autograd_and_existing_unsupported_boundaries_are_preserved(self):
        scalar = torch.tensor(0.5, requires_grad=True)
        scalar_output = functional.sigmoid(input=scalar)
        self.assertTrue(scalar_output.requires_grad)
        self.assertFalse(scalar_output.is_leaf)
        scalar_output.backward()
        self.assertEqual(self.tensor_bits(scalar.grad).item(), 0x3E70_A4D0)

        leaf = torch.tensor(
            [[-2.0, -0.0, 1.0], [2.0, 4.0, 8.0]], requires_grad=True
        )
        source = leaf.transpose(0, 1)[1]
        for call in (source.sigmoid, lambda: functional.sigmoid(source)):
            with self.subTest(call=call):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^sigmoid\(\): autograd recording is not supported$",
                ):
                    call()

        with torch.no_grad():
            actual = functional.sigmoid(source)
            expected = source.sigmoid()
        self.assert_tensor_matches(actual, expected, source, case="no_grad")

        detached = source.detach()
        self.assert_tensor_matches(
            functional.sigmoid(detached),
            detached.sigmoid(),
            detached,
            case="detached",
        )

    def test_argument_receiver_and_scope_errors(self):
        source = torch.tensor([0.5])
        cases = (
            (
                lambda: functional.sigmoid(),
                TypeError,
                "sigmoid() missing 1 required positional argument: 'input'",
            ),
            (
                lambda: functional.sigmoid(source, source),
                TypeError,
                "sigmoid() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: functional.sigmoid(source, input=source),
                TypeError,
                "sigmoid() got multiple values for argument 'input'",
            ),
            (
                lambda: functional.sigmoid(source, out=None),
                TypeError,
                "sigmoid() got an unexpected keyword argument 'out'",
            ),
            (
                lambda: functional.sigmoid(1),
                AttributeError,
                "'int' object has no attribute 'sigmoid'",
            ),
        )
        for case, (call, error_type, message) in enumerate(cases):
            with self.subTest(case=case):
                with self.assertRaises(error_type) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

        expected_error = ValueError("receiver failed")

        class RaisingReceiver:
            def sigmoid(self):
                raise expected_error

        with self.assertRaises(ValueError) as raised:
            functional.sigmoid(RaisingReceiver())
        self.assertIs(raised.exception, expected_error)

        class NonCallableReceiver:
            sigmoid = 1

        with self.assertRaisesRegex(TypeError, "^'int' object is not callable$"):
            functional.sigmoid(NonCallableReceiver())

        self.assertFalse(hasattr(torch, "sigmoid"))
        self.assertFalse(hasattr(nn, "Sigmoid"))
        self.assertFalse(hasattr(torch.Tensor, "sigmoid_"))
        self.assertFalse(hasattr(functional, "sigmoid_"))


if __name__ == "__main__":
    unittest.main()
