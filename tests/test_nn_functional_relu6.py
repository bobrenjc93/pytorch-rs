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


FUNCTION_DOC = r"""relu6(input, inplace=False) -> Tensor

    Applies the element-wise function :math:`\text{ReLU6}(x) = \min(\max(0,x), 6)`.

    See :class:`~torch.nn.ReLU6` for more details.
    """

if sys.version_info >= (3, 13):
    FUNCTION_DOC = (
        "relu6(input, inplace=False) -> Tensor\n\n"
        "Applies the element-wise function "
        r":math:`\text{ReLU6}(x) = \min(\max(0,x), 6)`."
        "\n\n"
        "See :class:`~torch.nn.ReLU6` for more details.\n"
    )


class FunctionalRelu6Tests(unittest.TestCase):
    @staticmethod
    def tensor_bits(tensor):
        return np.asarray(tensor).reshape(-1).view(np.uint32)

    @classmethod
    def tensor_state(cls, tensor):
        return (
            tensor.shape,
            tensor.stride(),
            tensor.storage_offset(),
            tensor.data_ptr(),
            tensor.requires_grad,
            tensor.is_leaf,
            cls.tensor_bits(tensor).copy(),
        )

    @staticmethod
    def expected_relu6_bits(bits):
        expected = []
        for raw_bits in bits:
            value_bits = int(raw_bits)
            magnitude = value_bits & 0x7FFF_FFFF
            if magnitude > 0x7F80_0000:
                expected.append(value_bits)
            elif value_bits & 0x8000_0000 and magnitude != 0:
                expected.append(0x0000_0000)
            elif magnitude > 0x40C0_0000:
                expected.append(0x40C0_0000)
            else:
                expected.append(value_bits)
        return np.asarray(expected, dtype=np.uint32)

    @staticmethod
    def layout_cases():
        base = torch.tensor(
            np.linspace(-3.0, 8.0, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist()
        )
        mixed_singleton = torch.tensor(
            np.linspace(-1.0, 7.0, 6, dtype=np.float32)
            .reshape(3, 1, 2)
            .tolist()
        ).permute(2, 1, 0)
        channels_last = torch.tensor(
            np.linspace(-4.0, 9.0, 120, dtype=np.float32)
            .reshape(2, 3, 4, 5)
            .tolist()
        ).contiguous(memory_format=torch.channels_last)
        return (
            ("scalar", torch.tensor(-0.0), ()),
            (
                "empty",
                torch.zeros((2, 0, 3)).transpose(0, 2)[1],
                (3, 3),
            ),
            ("offset", base[1], (4, 1)),
            ("strided", base.transpose(0, 2)[1], (1, 3)),
            ("mixed singleton", mixed_singleton, (1, 2, 2)),
            ("channels last", channels_last, (60, 1, 15, 3)),
        )

    def assert_relu6_result(self, actual, source, expected_stride, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, source.shape)
            self.assertEqual(actual.stride(), expected_stride)
            self.assertEqual(actual.storage_offset(), 0)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
            self.assertFalse(actual.requires_grad)
            self.assertTrue(actual.is_leaf)
            self.assertIsNot(actual, source)
            self.assertFalse(actual.is_set_to(source))
            if source.numel():
                self.assertNotEqual(actual.data_ptr(), source.data_ptr())
            if len(source.shape) == 4:
                self.assertTrue(
                    actual.is_contiguous(memory_format=torch.channels_last)
                )

        np.testing.assert_array_equal(
            self.tensor_bits(actual),
            self.expected_relu6_bits(self.tensor_bits(source)),
        )

    def test_import_signature_documentation_copy_and_pickle(self):
        imported_nn = importlib.import_module("torch_rs.nn")
        imported_functional = importlib.import_module("torch_rs.nn.functional")
        from torch_rs.nn import functional as from_nn
        from torch_rs.nn.functional import relu6

        self.assertIs(torch.nn, nn)
        self.assertIs(nn, imported_nn)
        self.assertIs(nn.functional, functional)
        self.assertIs(functional, imported_functional)
        self.assertIs(from_nn, functional)
        self.assertIs(relu6, functional.relu6)
        self.assertFalse(hasattr(nn, "__all__"))
        self.assertFalse(hasattr(functional, "__all__"))

        function = functional.relu6
        signature = inspect.signature(function)
        parameters = tuple(signature.parameters.values())
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "relu6")
        self.assertEqual(function.__qualname__, "relu6")
        self.assertEqual(function.__module__, "torch_rs.nn.functional")
        self.assertEqual(function.__defaults__, (False,))
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(
            function.__annotations__,
            {"input": torch.Tensor, "inplace": bool, "return": torch.Tensor},
        )
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertEqual(
            str(signature),
            "(input: torch_rs.Tensor, inplace: bool = False) -> torch_rs.Tensor",
        )
        self.assertEqual(tuple(signature.parameters), ("input", "inplace"))
        self.assertIs(parameters[0].annotation, torch.Tensor)
        self.assertIs(parameters[1].annotation, bool)
        self.assertIs(parameters[1].default, False)
        self.assertIs(signature.return_annotation, torch.Tensor)

        wildcard = {}
        exec("from torch_rs.nn.functional import *", wildcard)
        self.assertIs(wildcard["relu6"], function)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

    def test_supported_layouts_values_storage_and_nonmutation(self):
        for case, source, expected_stride in self.layout_cases():
            before = self.tensor_state(source)
            calls = (
                lambda: functional.relu6(source),
                lambda: functional.relu6(source, False),
                lambda: functional.relu6(input=source, inplace=False),
            )
            for form, call in enumerate(calls):
                actual = call()
                self.assert_relu6_result(
                    actual,
                    source,
                    expected_stride,
                    case=(case, form),
                )
            after = self.tensor_state(source)
            with self.subTest(case=case, nonmutation=True):
                self.assertEqual(after[:-1], before[:-1])
                np.testing.assert_array_equal(after[-1], before[-1])

    def test_float32_boundary_infinity_and_nan_bits(self):
        input_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x007F_FFFF,
                0x807F_FFFF,
                0x0080_0000,
                0x8080_0000,
                0x3F7F_FFFF,
                0x3F80_0000,
                0x40BF_FFFE,
                0x40BF_FFFF,
                0x40C0_0000,
                0x40C0_0001,
                0x40C0_0002,
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
        source = torch.tensor(memoryview(input_bits.view(np.float32)))
        actual = functional.relu6(source)
        np.testing.assert_array_equal(
            self.tensor_bits(actual),
            self.expected_relu6_bits(self.tensor_bits(source)),
        )

    def test_every_call_returns_fresh_independent_storage(self):
        for case, source, _ in self.layout_cases():
            first = functional.relu6(source)
            second = functional.relu6(source)
            with self.subTest(case=case):
                self.assertIsNot(first, second)
                self.assertFalse(first.is_set_to(second))
                self.assertFalse(first.is_set_to(source))
                if first.numel():
                    self.assertNotEqual(first.data_ptr(), second.data_ptr())
                    self.assertNotEqual(first.data_ptr(), source.data_ptr())

    def test_overrides_and_modes_observe_the_public_function_and_inplace(self):
        marker = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        for inplace in (False, True):
            value = Override()
            result = functional.relu6(input=value, inplace=inplace)
            self.assertIs(result, marker)
            function, dispatch_types, args, kwargs = Override.calls.pop(0)
            self.assertIs(function, functional.relu6)
            self.assertEqual(dispatch_types, (Override,))
            self.assertEqual(args, (value,))
            self.assertEqual(kwargs, {"inplace": inplace})

        source = torch.tensor([0.5], requires_grad=True)

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        for inplace in (False, True):
            mode = RecordingMode()
            with mode:
                result = functional.relu6(source, inplace=inplace)
            self.assertIs(result, marker)
            self.assertEqual(len(mode.calls), 1)
            function, dispatch_types, args, kwargs = mode.calls[0]
            self.assertIs(function, functional.relu6)
            self.assertEqual(dispatch_types, (torch.Tensor,))
            self.assertEqual(args, (source,))
            self.assertEqual(kwargs, {"inplace": inplace})

    def test_active_autograd_is_rejected_before_native_allocation(self):
        leaf = torch.tensor(
            [[-2.0, -0.0, 1.0], [2.0, 6.0, 8.0]], requires_grad=True
        )
        for case, source in (
            ("scalar", torch.tensor(0.5, requires_grad=True)),
            ("strided view", leaf.transpose(0, 1)[1]),
        ):
            before = self.tensor_state(source)
            with self.subTest(case=case):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^relu6\(\): autograd recording is not supported$",
                ):
                    functional.relu6(source)
                after = self.tensor_state(source)
                self.assertEqual(after[:-1], before[:-1])
                np.testing.assert_array_equal(after[-1], before[-1])
                self.assertIsNone(leaf.grad)

        extreme = torch.zeros((0,), requires_grad=True).reshape(
            (0, sys.maxsize, 3)
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"^relu6\(\): autograd recording is not supported$",
        ):
            functional.relu6(extreme)
        with torch.no_grad():
            output = functional.relu6(extreme)
        self.assertEqual(output.shape, extreme.shape)
        self.assertEqual(output.stride(), extreme.stride())
        self.assertEqual(output.numel(), 0)
        self.assertFalse(output.is_set_to(extreme))

    def test_no_grad_and_detached_inputs_use_the_inference_path(self):
        leaf = torch.tensor(
            [[-2.0, -0.0, 1.0], [2.0, 6.0, 8.0]], requires_grad=True
        )
        source = leaf.transpose(0, 1)[1]

        with torch.no_grad():
            actual = functional.relu6(source)
        self.assert_relu6_result(actual, source, (1,), case="no_grad")

        detached = source.detach()
        actual = functional.relu6(detached)
        self.assert_relu6_result(actual, detached, (1,), case="detached")
        self.assertIsNone(leaf.grad)

    def test_inplace_true_is_rejected_before_allocation_and_does_not_mutate(self):
        leaf = torch.tensor(
            [[9.0, 9.0, 9.0], [-1.0, 2.0, -0.0]], requires_grad=True
        )
        source = leaf[1]
        before = self.tensor_state(source)

        for call in (
            lambda: functional.relu6(source, True),
            lambda: functional.relu6(input=source, inplace=True),
        ):
            with self.assertRaisesRegex(
                NotImplementedError,
                r"^torch_rs\.nn\.functional\.relu6 does not support inplace=True$",
            ):
                call()

        after = self.tensor_state(source)
        self.assertEqual(after[:-1], before[:-1])
        np.testing.assert_array_equal(after[-1], before[-1])
        self.assertIsNone(leaf.grad)

        extreme = torch.zeros((0,), requires_grad=True).reshape(
            (0, sys.maxsize, 3)
        )
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^torch_rs\.nn\.functional\.relu6 does not support inplace=True$",
        ):
            functional.relu6(extreme, inplace=True)

    def test_argument_errors_and_module_boundary(self):
        source = torch.tensor([0.5])
        cases = (
            (
                lambda: functional.relu6(),
                TypeError,
                "relu6() missing 1 required positional argument: 'input'",
            ),
            (
                lambda: functional.relu6(source, False, None),
                TypeError,
                "relu6() takes from 1 to 2 positional arguments but 3 were given",
            ),
            (
                lambda: functional.relu6(source, input=source),
                TypeError,
                "relu6() got multiple values for argument 'input'",
            ),
            (
                lambda: functional.relu6(source, out=None),
                TypeError,
                "relu6() got an unexpected keyword argument 'out'",
            ),
            (
                lambda: functional.relu6(1),
                TypeError,
                "relu6(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: functional.relu6(1, inplace=True),
                TypeError,
                "relu6_(): argument 'input' (position 1) must be Tensor, not int",
            ),
        )
        for case, (call, error_type, message) in enumerate(cases):
            with self.subTest(case=case):
                with self.assertRaises(error_type) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

        self.assertFalse(hasattr(torch, "relu6"))
        self.assertFalse(hasattr(torch, "_nn_functional_relu6"))
        self.assertFalse(hasattr(nn, "ReLU6"))
        self.assertFalse(hasattr(torch.Tensor, "relu6"))
        self.assertFalse(hasattr(functional, "relu6_"))


if __name__ == "__main__":
    unittest.main()
