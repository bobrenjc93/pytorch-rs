import copy
import importlib
import inspect
import operator
import pickle
import re
import sys
import types
import unittest
from multiprocessing.reduction import ForkingPickler

import numpy as np
import torch_rs as torch


class TensorDivideMethodTests(unittest.TestCase):
    def assert_tensor_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual).reshape(-1).view(np.uint32),
                np.asarray(expected).reshape(-1).view(np.uint32),
            )

    def assert_alias_matches_true_division(self, input, other, *, case):
        expected = input / other
        for name in ("div", "divide"):
            method = getattr(input, name)
            self.assert_tensor_matches(
                method(other), expected, case=(case, name, "positional")
            )
            self.assert_tensor_matches(
                method(other=other), expected, case=(case, name, "other")
            )
            self.assert_tensor_matches(
                method(x2=other), expected, case=(case, name, "x2")
            )
            self.assert_tensor_matches(
                method(other=other, rounding_mode=None),
                expected,
                case=(case, name, "rounding_mode=None"),
            )
            self.assert_tensor_matches(
                method(x2=other, rounding_mode=None),
                expected,
                case=(case, name, "x2 rounding_mode=None"),
            )

    def test_tensor_scalar_broadcast_empty_offset_and_special_values(self):
        left = torch.tensor([[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]).transpose(
            0, 2
        )
        right = torch.tensor([[2.0], [3.0], [4.0]])
        self.assert_alias_matches_true_division(
            left, right, case="tensor broadcasting"
        )

        offset_view = left[1]
        for scalar in (
            True,
            -2,
            2.5,
            np.bool_(True),
            np.int64(3),
            np.float32(-0.0),
        ):
            self.assert_alias_matches_true_division(
                offset_view, scalar, case=("scalar", type(scalar).__name__, scalar)
            )

        empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        broadcast = torch.ones((1, 1, 2))
        self.assert_alias_matches_true_division(
            empty, broadcast, case="strided broadcast empty"
        )

        special_bits = np.asarray(
            (0x0000_0000, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC1_2345),
            dtype=np.uint32,
        )
        special = torch.tensor(memoryview(special_bits.view(np.float32)))
        self.assert_alias_matches_true_division(
            special, -0.0, case="signed zero and non-finites"
        )

    def test_no_grad_reuses_true_division_semantics(self):
        grad_left = torch.tensor([[1.0, 2.0]], requires_grad=True)
        grad_right = torch.tensor([[3.0], [4.0]], requires_grad=True)
        with torch.no_grad():
            tensor_output = grad_left.transpose(0, 1).div(
                grad_right.transpose(0, 1)
            )
            scalar_output = grad_left.divide(other=2.0)
        self.assertFalse(tensor_output.requires_grad)
        self.assertFalse(scalar_output.requires_grad)

        self.assert_tensor_matches(
            grad_left.div(grad_right.transpose(0, 1)),
            grad_left / grad_right.transpose(0, 1),
            case="tracked tensor metadata",
        )

    def test_descriptor_metadata_unbound_call_copy_pickle_and_reload(self):
        tensor = torch.tensor([1.0, 2.0])
        docs = {
            "div": "\ndiv(value, *, rounding_mode=None) -> Tensor\n\nSee :func:`torch.div`\n",
            "divide": "\ndivide(value, *, rounding_mode=None) -> Tensor\n\nSee :func:`torch.divide`\n",
        }

        for name in ("div", "divide"):
            descriptor = inspect.getattr_static(torch.Tensor, name)
            bound = getattr(tensor, name)
            with self.subTest(name=name, metadata=True):
                self.assertIs(type(descriptor), types.MethodDescriptorType)
                self.assertIs(type(bound), types.BuiltinMethodType)
                self.assertEqual(descriptor.__name__, name)
                self.assertEqual(bound.__name__, name)
                self.assertEqual(descriptor.__qualname__, f"TensorBase.{name}")
                self.assertEqual(bound.__qualname__, f"Tensor.{name}")
                self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
                self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
                self.assertIsNone(descriptor.__text_signature__)
                self.assertIsNone(bound.__text_signature__)
                self.assertEqual(descriptor.__doc__, docs[name])
                with self.assertRaises(ValueError):
                    inspect.signature(descriptor)
                with self.assertRaises(ValueError):
                    inspect.signature(bound)

            self.assert_tensor_matches(
                descriptor(tensor, other=tensor),
                tensor / tensor,
                case=(name, "unbound"),
            )
            self.assert_tensor_matches(
                descriptor(tensor, x2=2.0, rounding_mode=None),
                tensor / 2.0,
                case=(name, "unbound x2"),
            )

            self.assertIs(copy.copy(descriptor), descriptor)
            self.assertIs(copy.deepcopy(descriptor), descriptor)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(name=name, protocol=protocol, pickler="pickle"):
                    self.assertIs(
                        pickle.loads(pickle.dumps(descriptor, protocol)),
                        descriptor,
                    )
                with self.subTest(
                    name=name, protocol=protocol, pickler="ForkingPickler"
                ):
                    self.assertIs(
                        pickle.loads(ForkingPickler.dumps(descriptor, protocol)),
                        descriptor,
                    )
            self.assertIs(copy.copy(bound), bound)
            self.assertIs(copy.deepcopy(bound), bound)

        reloaded = importlib.reload(torch)
        for name in ("div", "divide"):
            self.assertIs(
                inspect.getattr_static(reloaded.Tensor, name),
                inspect.getattr_static(torch.Tensor, name),
            )

    def test_descriptor_reducer_survives_package_reinitialization(self):
        original_modules = {
            name: module
            for name, module in tuple(sys.modules.items())
            if name == "torch_rs" or name.startswith("torch_rs.")
        }
        try:
            for name in original_modules:
                sys.modules.pop(name, None)
            reinitialized = importlib.import_module("torch_rs")
            for name in ("div", "divide"):
                descriptor = inspect.getattr_static(reinitialized.Tensor, name)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    with self.subTest(
                        name=name, protocol=protocol, package="reinitialized"
                    ):
                        self.assertIs(
                            pickle.loads(pickle.dumps(descriptor, protocol)),
                            descriptor,
                        )
                        self.assertIs(
                            pickle.loads(ForkingPickler.dumps(descriptor, protocol)),
                            descriptor,
                        )
        finally:
            for name in tuple(sys.modules):
                if name == "torch_rs" or name.startswith("torch_rs."):
                    sys.modules.pop(name, None)
            sys.modules.update(original_modules)

        for name in ("div", "divide"):
            descriptor = inspect.getattr_static(torch.Tensor, name)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(name=name, protocol=protocol, package="restored"):
                    self.assertIs(
                        pickle.loads(pickle.dumps(descriptor, protocol)),
                        descriptor,
                    )
                    self.assertIs(
                        pickle.loads(ForkingPickler.dumps(descriptor, protocol)),
                        descriptor,
                    )

    def test_argument_errors_and_rejected_extensions(self):
        tensor = torch.tensor([1.0])
        div_overloads = (
            "but expected one of:\n"
            " * (Tensor other)\n"
            " * (Tensor other, *, str rounding_mode)\n"
            " * (Number other, *, str rounding_mode)\n"
        )
        divide_overloads = (
            "but expected one of:\n"
            " * (Tensor other)\n"
            " * (Tensor other, *, str rounding_mode)\n"
            " * (Number other)\n"
            " * (Number other, *, str rounding_mode)\n"
        )
        cases = (
            (
                lambda: tensor.div(),
                "div() received an invalid combination of arguments - got (), "
                f"{div_overloads}",
            ),
            (
                lambda: tensor.divide(),
                "divide() received an invalid combination of arguments - got (), "
                f"{divide_overloads}",
            ),
            (
                lambda: tensor.div(tensor, tensor),
                "div() received an invalid combination of arguments - got "
                f"(Tensor, Tensor), {div_overloads}",
            ),
            (
                lambda: tensor.divide(tensor, tensor),
                "divide() received an invalid combination of arguments - got "
                f"(Tensor, Tensor), {divide_overloads}",
            ),
            (
                lambda: tensor.div(tensor, out=tensor),
                "div() received an invalid combination of arguments - got "
                f"(Tensor, out=Tensor), {div_overloads}",
            ),
            (
                lambda: tensor.divide(tensor, out=tensor),
                "divide() received an invalid combination of arguments - got "
                f"(Tensor, out=Tensor), {divide_overloads}",
            ),
            (
                lambda: tensor.div(dtype=torch.float32),
                'div() missing 1 required positional arguments: "other"',
            ),
            (
                lambda: tensor.divide(device=torch.device("cpu")),
                "divide() received an invalid combination of arguments - got "
                "(device=torch.device, ), but expected one of:\n"
                " * (Tensor other)\n"
                "      didn't match because some of the keywords were incorrect: device\n"
                " * (Tensor other, *, str rounding_mode)\n"
                " * (Number other)\n"
                "      didn't match because some of the keywords were incorrect: device\n"
                " * (Number other, *, str rounding_mode)\n",
            ),
            (
                lambda: tensor.div([]),
                "div(): argument 'other' (position 1) must be Tensor, not list",
            ),
            (
                lambda: tensor.div(other=None),
                "div(): argument 'other' must be Tensor, not NoneType",
            ),
            (
                lambda: tensor.divide([]),
                "divide() received an invalid combination of arguments - got "
                "(list), but expected one of:\n"
                " * (Tensor other)\n"
                "      didn't match because some of the arguments have invalid types: "
                "(!list of []!)\n"
                " * (Tensor other, *, str rounding_mode)\n"
                " * (Number other)\n"
                "      didn't match because some of the arguments have invalid types: "
                "(!list of []!)\n"
                " * (Number other, *, str rounding_mode)\n",
            ),
            (lambda: tensor.div(np.uint64(2**63)), "an integer is required"),
            (lambda: tensor.divide(np.uint64(2**63)), "an integer is required"),
            (lambda: tensor.div(2**64), "int too big to convert"),
            (lambda: tensor.divide(2**64), "int too big to convert"),
            (
                lambda: tensor.div(-(2**63) - 1),
                "can't convert negative int to unsigned",
            ),
            (
                lambda: tensor.divide(-(2**63) - 1),
                "can't convert negative int to unsigned",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(Exception, f"^{re.escape(message)}$"):
                    call()

        for name in ("div", "divide"):
            with self.subTest(name=name, rounding_mode="floor"):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    rf"^{name}\(\): rounding_mode is not supported; only None is implemented$",
                ):
                    getattr(tensor, name)(2.0, rounding_mode="floor")
            with self.subTest(name=name, rounding_mode="trunc"):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    rf"^{name}\(\): rounding_mode is not supported; only None is implemented$",
                ):
                    getattr(tensor, name)(tensor, rounding_mode="trunc")

    def test_modes_subclass_like_operands_and_inplace_forms_are_rejected(self):
        tensor = torch.tensor([1.0, 2.0])
        calls = []

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                calls.append((func, types, args, kwargs))
                return "mode result"

        for name in ("div", "divide"):
            with self.subTest(name=name, mode=True):
                with RecordingMode():
                    with self.assertRaisesRegex(
                        TypeError,
                        rf"^{name}\(\) does not support an active TorchFunctionMode$",
                    ):
                        getattr(tensor, name)(2.0)
                self.assertEqual(calls, [])

        class TensorLike:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                calls.append((func, types, args, kwargs))
                return "override result"

        with self.assertRaisesRegex(
            TypeError, r"^div\(\): argument 'other' \(position 1\) must be Tensor"
        ):
            tensor.div(TensorLike())
        with self.assertRaisesRegex(
            TypeError,
            r"^divide\(\) received an invalid combination of arguments",
        ):
            tensor.divide(TensorLike())
        self.assertEqual(calls, [])

        self.assertFalse(hasattr(torch.Tensor, "div_"))
        self.assertFalse(hasattr(torch.Tensor, "divide_"))
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^Tensor\.__itruediv__\(\): in-place division is not supported$",
        ):
            operator.itruediv(tensor, 2.0)


if __name__ == "__main__":
    unittest.main()
