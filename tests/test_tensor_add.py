import inspect
import re
import types
import unittest

import numpy as np
import torch_rs as torch


METHOD_DOC = (
    "\nadd(other, *, alpha=1) -> Tensor\n\n"
    "Add a scalar or tensor to :attr:`self` tensor. If both :attr:`alpha`\n"
    "and :attr:`other` are specified, each element of :attr:`other` is scaled by\n"
    ":attr:`alpha` before being used.\n\n"
    "When :attr:`other` is a tensor, the shape of :attr:`other` must be\n"
    ":ref:`broadcastable <broadcasting-semantics>` with the shape of the underlying\n"
    "tensor\n\n"
    "See :func:`torch.add`\n"
)


class TensorAddTests(unittest.TestCase):
    def assert_tensor_matches(self, actual, expected, source, *, case):
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
            if source.numel():
                self.assertNotEqual(actual.data_ptr(), source.data_ptr())
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual, dtype=np.float32).reshape(-1).view(np.uint32),
                np.asarray(expected, dtype=np.float32).reshape(-1).view(np.uint32),
            )

    @staticmethod
    def make_cases():
        base = torch.tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist())
        strided = base.transpose(0, 2)
        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
                0x7F81_2345,
                0xFF85_4321,
            ),
            dtype=np.uint32,
        )
        return (
            ("scalar", torch.tensor(-0.0)),
            ("empty offset", torch.zeros((2, 0, 3)).transpose(0, 2)[1]),
            ("empty singleton trailing", torch.zeros((0, 1))),
            ("empty singleton middle", torch.zeros((0, 1, 2))),
            ("offset", strided[1]),
            ("noncontiguous", strided),
            ("IEEE edges", torch.tensor(memoryview(special_bits.view(np.float32)))),
        )

    def test_scalar_values_layouts_and_fresh_storage_match_operator(self):
        for case, source in self.make_cases():
            for scalar in (2.5, np.float32(-0.0)):
                expected = source + scalar
                self.assert_tensor_matches(
                    source.add(scalar), expected, source, case=(case, "positional", scalar)
                )
                self.assert_tensor_matches(
                    source.add(other=scalar),
                    expected,
                    source,
                    case=(case, "keyword", scalar),
                )

    def test_real_scalar_type_forms_match_operator(self):
        source = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        ).transpose(0, 2)[1]
        scalars = (
            True,
            False,
            -2,
            0,
            2.5,
            np.bool_(True),
            np.bool_(False),
            np.int64(3),
            np.uint64(3),
            np.float32(-0.0),
            np.float64(1.25),
        )
        for scalar in scalars:
            with self.subTest(scalar=scalar, type=type(scalar).__name__):
                self.assert_tensor_matches(
                    source.add(scalar), source + scalar, source, case=scalar
                )

    def test_default_alpha_forms_match_operator(self):
        source = torch.tensor([1.0, -2.0, 3.5])
        alpha_values = (
            1,
            1.0,
            1.0000000001,
            np.bool_(True),
            np.int64(1),
            np.uint64(1),
            np.float32(1.0),
            np.float64(1.0),
        )
        for alpha in alpha_values:
            with self.subTest(alpha=alpha, type=type(alpha).__name__):
                self.assert_tensor_matches(
                    source.add(2.0, alpha=alpha),
                    source + 2.0,
                    source,
                    case=alpha,
                )
        self.assert_tensor_matches(
            source.add(x2=2.0, alpha=1),
            source + 2.0,
            source,
            case="legacy x2 alias",
        )

    def test_autograd_and_no_grad_match_operator(self):
        method_leaf = torch.tensor([2.0, -3.0], requires_grad=True)
        operator_leaf = torch.tensor([2.0, -3.0], requires_grad=True)
        method_output = method_leaf.add(4.0)
        operator_output = operator_leaf + 4.0

        self.assert_tensor_matches(
            method_output, operator_output, method_leaf, case="tracked output"
        )
        method_output.sum().backward()
        operator_output.sum().backward()
        self.assert_tensor_matches(
            method_leaf.grad, operator_leaf.grad, method_leaf, case="gradient"
        )

        no_grad_leaf = torch.tensor([[1.0, 2.0]], requires_grad=True)
        with torch.no_grad():
            no_grad_output = no_grad_leaf.transpose(0, 1).add(other=2.0)
        self.assertFalse(no_grad_output.requires_grad)
        self.assertTrue(no_grad_leaf.add(2.0).requires_grad)

    def test_descriptor_metadata_unbound_call_and_argument_errors(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "add")
        bound = tensor.add

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(repr(descriptor), "<method 'add' of 'torch._C.TensorBase' objects>")
        self.assertEqual(descriptor.__name__, "add")
        self.assertEqual(bound.__name__, "add")
        self.assertEqual(descriptor.__qualname__, "TensorBase.add")
        self.assertEqual(bound.__qualname__, "Tensor.add")
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        self.assertIsNone(descriptor.__text_signature__)
        self.assertIsNone(bound.__text_signature__)
        self.assertEqual(descriptor.__doc__, METHOD_DOC)
        for callable_object in (descriptor, bound):
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)
        self.assert_tensor_matches(
            descriptor(tensor, other=2.0), tensor + 2.0, tensor, case="unbound call"
        )

        unsupported_other = (
            "Tensor.add() only supports scalar 'other'; Tensor operands are not supported"
        )
        unsupported_alpha = "Tensor.add() only supports the default alpha=1"
        cases = (
            (
                lambda: tensor.add(),
                TypeError,
                "add() received an invalid combination of arguments - got (), but expected "
                "(Tensor other, *, Number alpha = 1)",
            ),
            (
                lambda: tensor.add(1, 2),
                TypeError,
                "add() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: tensor.add(1, 2, 3),
                TypeError,
                "add() takes 1 positional argument but 3 were given",
            ),
            (
                lambda: tensor.add(1, other=2),
                TypeError,
                "add() got multiple values for argument 'other'",
            ),
            (
                lambda: tensor.add(1, out=tensor),
                TypeError,
                "add() got an unexpected keyword argument 'out'",
            ),
            (
                lambda: tensor.add(wat=2),
                TypeError,
                'add() missing 1 required positional arguments: "other"',
            ),
            (
                lambda: tensor.add([]),
                TypeError,
                "add(): argument 'other' (position 1) must be Number, not list",
            ),
            (
                lambda: tensor.add(other=None),
                TypeError,
                "add(): argument 'other' must be Number, not NoneType",
            ),
            (
                lambda: tensor.add(torch.tensor([1.0])),
                NotImplementedError,
                unsupported_other,
            ),
            (
                lambda: tensor.add(other=torch.tensor([1.0])),
                NotImplementedError,
                unsupported_other,
            ),
            (
                lambda: tensor.add(1.0, alpha=2),
                NotImplementedError,
                unsupported_alpha,
            ),
            (
                lambda: tensor.add(1.0, alpha=0),
                NotImplementedError,
                unsupported_alpha,
            ),
            (
                lambda: tensor.add(1.0, alpha=np.float32(1.0000001)),
                NotImplementedError,
                unsupported_alpha,
            ),
            (
                lambda: tensor.add(1.0, alpha=True),
                RuntimeError,
                "Boolean alpha only supported for Boolean results.",
            ),
            (
                lambda: tensor.add(1.0, alpha=[]),
                TypeError,
                "add(): argument 'alpha' must be Number, not list",
            ),
            (
                lambda: tensor.add(np.uint64(2**63)),
                TypeError,
                "an integer is required",
            ),
            (
                lambda: tensor.add(2**64),
                OverflowError,
                "int too big to convert",
            ),
            (
                lambda: tensor.add(-(2**63) - 1),
                OverflowError,
                "can't convert negative int to unsigned",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error_type, f"^{re.escape(message)}$"):
                    call()

        descriptor_cases = (
            (
                lambda: descriptor(),
                "unbound method TensorBase.add() needs an argument",
            ),
            (
                lambda: descriptor(1),
                "descriptor 'add' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.add() needs an argument",
            ),
            (
                lambda: descriptor(tensor),
                "add() received an invalid combination of arguments - got (), but expected "
                "(Tensor other, *, Number alpha = 1)",
            ),
        )
        for call, message in descriptor_cases:
            with self.subTest(descriptor_error=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

    def test_unsupported_top_level_tensor_operands_and_in_place_are_absent(self):
        tensor = torch.tensor([1.0])
        self.assertFalse(hasattr(torch, "add"))
        self.assertNotIn("add", torch.__all__)
        self.assertFalse(hasattr(torch.Tensor, "add_"))
        self.assertFalse(hasattr(tensor, "add_"))


if __name__ == "__main__":
    unittest.main()
