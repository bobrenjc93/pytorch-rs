import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


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


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorAddReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("Tensor.add differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def assert_matches(self, actual, expected, source, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, tuple(expected.shape))
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
            actual_bits = np.asarray(actual, dtype=np.float32).reshape(-1).view(np.uint32)
            expected_bits = expected.detach().cpu().numpy().reshape(-1).view(np.uint32)
            np.testing.assert_array_equal(actual_bits, expected_bits)

    @staticmethod
    def make_cases(module):
        base = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
        )
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
            ("scalar", module.tensor(-0.0, dtype=module.float32)),
            (
                "empty offset",
                module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
            ),
            ("empty singleton trailing", module.zeros((0, 1), dtype=module.float32)),
            ("empty singleton middle", module.zeros((0, 1, 2), dtype=module.float32)),
            ("offset", strided[1]),
            ("noncontiguous", strided),
            (
                "IEEE edges",
                module.tensor(memoryview(special_bits.view(np.float32))),
            ),
        )

    def test_scalar_values_layouts_and_storage_match_pytorch_2_13(self):
        actual_cases = self.make_cases(torch)
        expected_cases = self.make_cases(reference_torch)
        for (case, actual_source), (_, expected_source) in zip(
            actual_cases, expected_cases, strict=True
        ):
            for scalar in (2.5, np.float32(-0.0)):
                with self.subTest(case=case, scalar=scalar):
                    self.assert_matches(
                        actual_source.add(scalar),
                        expected_source.add(scalar),
                        actual_source,
                        case=(case, "positional", scalar),
                    )
                    self.assert_matches(
                        actual_source.add(other=scalar),
                        expected_source.add(other=scalar),
                        actual_source,
                        case=(case, "keyword", scalar),
                    )

    def test_real_scalar_and_default_alpha_forms_match_pytorch_2_13(self):
        actual = torch.tensor([1.0, -2.0, 3.5])
        expected = reference_torch.tensor([1.0, -2.0, 3.5])
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
                self.assert_matches(
                    actual.add(scalar),
                    expected.add(scalar),
                    actual,
                    case=("scalar", scalar),
                )

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
                self.assert_matches(
                    actual.add(2.0, alpha=alpha),
                    expected.add(2.0, alpha=alpha),
                    actual,
                    case=("alpha", alpha),
                )
        self.assert_matches(
            actual.add(x2=2.0, alpha=1),
            expected.add(x2=2.0, alpha=1),
            actual,
            case="legacy x2 alias",
        )

    def test_autograd_and_no_grad_match_pytorch_2_13(self):
        actual_leaf = torch.tensor([2.0, -3.0], requires_grad=True)
        expected_leaf = reference_torch.tensor([2.0, -3.0], requires_grad=True)

        actual_output = actual_leaf.add(4.0)
        expected_output = expected_leaf.add(4.0)
        self.assert_matches(actual_output, expected_output, actual_leaf, case="tracked")
        actual_output.sum().backward()
        expected_output.sum().backward()
        self.assert_matches(
            actual_leaf.grad, expected_leaf.grad, actual_leaf, case="gradient"
        )

        actual_no_grad = torch.tensor([[1.0, 2.0]], requires_grad=True)
        expected_no_grad = reference_torch.tensor([[1.0, 2.0]], requires_grad=True)
        with torch.no_grad():
            actual_untracked = actual_no_grad.transpose(0, 1).add(other=2.0)
        with reference_torch.no_grad():
            expected_untracked = expected_no_grad.transpose(0, 1).add(other=2.0)
        self.assert_matches(
            actual_untracked, expected_untracked, actual_no_grad, case="no_grad"
        )
        self.assertTrue(actual_no_grad.add(2.0).requires_grad)
        self.assertTrue(expected_no_grad.add(2.0).requires_grad)

    def method_contract(self, module):
        tensor = module.tensor([1.0])
        descriptor = inspect.getattr_static(module.Tensor, "add")
        bound = tensor.add
        signature_errors = []
        for callable_object in (descriptor, bound):
            try:
                inspect.signature(callable_object)
            except Exception as error:
                message = re.sub(
                    r"torch_rs\.Tensor object", "Tensor object", str(error)
                )
                message = re.sub(r"0x[0-9a-f]+", "0x...", message)
                signature_errors.append((type(error).__name__, message))
            else:
                signature_errors.append(None)
        reduce_roundtrips = []
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            try:
                reduce_roundtrips.append(
                    pickle.loads(pickle.dumps(descriptor, protocol=protocol))
                    is descriptor
                )
            except Exception as error:
                reduce_roundtrips.append((type(error).__name__, str(error)))
        return {
            "descriptor_type": type(descriptor),
            "bound_type": type(bound),
            "repr": repr(descriptor),
            "name": descriptor.__name__,
            "bound_name": bound.__name__,
            "qualname": descriptor.__qualname__,
            "bound_qualname": bound.__qualname__,
            "objclass_name": descriptor.__objclass__.__name__,
            "objclass_module": descriptor.__objclass__.__module__.replace(
                "torch_rs._C", "torch._C"
            ),
            "has_module": hasattr(descriptor, "__module__"),
            "bound_has_module": hasattr(bound, "__module__"),
            "text_signature": descriptor.__text_signature__,
            "bound_text_signature": bound.__text_signature__,
            "doc": descriptor.__doc__,
            "signature_errors": tuple(signature_errors),
            "reduce_roundtrips": tuple(reduce_roundtrips),
        }

    def test_descriptor_metadata_matches_pytorch_2_13(self):
        actual = self.method_contract(torch)
        expected = self.method_contract(reference_torch)
        self.assertEqual(actual, expected)
        self.assertEqual(actual["doc"], METHOD_DOC)
        self.assert_matches(
            inspect.getattr_static(torch.Tensor, "add")(torch.tensor([1.0]), other=2.0),
            inspect.getattr_static(reference_torch.Tensor, "add")(
                reference_torch.tensor([1.0]), other=2.0
            ),
            torch.tensor([1.0]),
            case="unbound call",
        )

    def test_matching_error_behavior_for_supported_boundary_cases(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        actual_descriptor = inspect.getattr_static(torch.Tensor, "add")
        expected_descriptor = inspect.getattr_static(reference_torch.Tensor, "add")
        cases = (
            (lambda: actual_descriptor(), lambda: expected_descriptor()),
            (lambda: actual_descriptor(1), lambda: expected_descriptor(1)),
            (
                lambda: actual_descriptor(self=actual),
                lambda: expected_descriptor(self=expected),
            ),
            (lambda: actual_descriptor(actual), lambda: expected_descriptor(expected)),
            (lambda: actual.add(), lambda: expected.add()),
            (lambda: actual.add(2.0, out=None), lambda: expected.add(2.0, out=None)),
            (
                lambda: actual.add(2.0, alpha=True),
                lambda: expected.add(2.0, alpha=True),
            ),
            (
                lambda: actual.add(2.0, alpha=[]),
                lambda: expected.add(2.0, alpha=[]),
            ),
            (
                lambda: actual.add(2.0, alpha=np.uint64(2**63)),
                lambda: expected.add(2.0, alpha=np.uint64(2**63)),
            ),
            (
                lambda: actual.add(2.0, alpha=2**64),
                lambda: expected.add(2.0, alpha=2**64),
            ),
            (
                lambda: actual.add(2.0, alpha=-(2**63) - 1),
                lambda: expected.add(2.0, alpha=-(2**63) - 1),
            ),
            (
                lambda: actual.add(np.uint64(2**63)),
                lambda: expected.add(np.uint64(2**63)),
            ),
            (lambda: actual.add(2**64), lambda: expected.add(2**64)),
            (
                lambda: actual.add(-(2**63) - 1),
                lambda: expected.add(-(2**63) - 1),
            ),
        )
        for actual_call, expected_call in cases:
            with self.subTest(actual=actual_call):
                self.assert_error_matches(actual_call, expected_call)

    def test_unsupported_scope_remains_absent(self):
        self.assertFalse(hasattr(torch, "add"))
        self.assertNotIn("add", torch.__all__)
        self.assertFalse(hasattr(torch.Tensor, "add_"))


if __name__ == "__main__":
    unittest.main()
