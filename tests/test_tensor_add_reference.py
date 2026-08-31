import copy
import inspect
import pickle
import re
import types
import unittest
import warnings

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorAddMethodReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("Tensor.add differentials require pinned PyTorch 2.13.0")

    def assert_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
        with self.subTest(case=case, values=True):
            actual_bits = np.asarray(actual).reshape(-1).view(np.uint32)
            expected_bits = expected.detach().cpu().numpy().reshape(-1).view(np.uint32)
            np.testing.assert_array_equal(actual_bits, expected_bits)

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def test_values_layouts_ieee_empties_and_argument_forms_match_pytorch_2_13(self):
        actual_left = torch.tensor(
            [[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]
        ).transpose(0, 2)
        expected_left = reference_torch.tensor(
            [[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]
        ).transpose(0, 2)
        actual_right = torch.tensor([[2.0], [-0.0], [float("inf")]])
        expected_right = reference_torch.tensor([[2.0], [-0.0], [float("inf")]])

        calls = (
            (
                "positional tensors",
                lambda: actual_left.add(actual_right),
                lambda: expected_left.add(expected_right),
            ),
            (
                "other keyword",
                lambda: actual_left.add(other=actual_right),
                lambda: expected_left.add(other=expected_right),
            ),
            (
                "x2 alias",
                lambda: actual_left.add(x2=actual_right),
                lambda: expected_left.add(x2=expected_right),
            ),
            (
                "default alpha",
                lambda: actual_left.add(actual_right, alpha=1),
                lambda: expected_left.add(expected_right, alpha=1),
            ),
            (
                "default numpy alpha",
                lambda: actual_left.add(actual_right, alpha=np.int64(1)),
                lambda: expected_left.add(expected_right, alpha=np.int64(1)),
            ),
            (
                "default numpy bool alpha",
                lambda: actual_left.add(actual_right, alpha=np.bool_(True)),
                lambda: expected_left.add(expected_right, alpha=np.bool_(True)),
            ),
        )
        for case, actual_call, expected_call in calls:
            self.assert_matches(actual_call(), expected_call(), case=case)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            expected_legacy = expected_left.add(1, expected_right)
        self.assert_matches(
            actual_left.add(1, actual_right),
            expected_legacy,
            case="legacy positional default alpha tensor other",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            expected_legacy_scalar = expected_left.add(1, 2)
        legacy_calls = (
            (
                "two positional scalars",
                actual_left.add(1, 2),
                expected_legacy_scalar,
            ),
            (
                "keyword scalar other",
                actual_left.add(1, other=2),
                expected_left.add(1, other=2),
            ),
            (
                "keyword tensor other",
                actual_left.add(1, other=actual_right),
                expected_left.add(1, other=expected_right),
            ),
            (
                "x2 tensor other",
                actual_left.add(1, x2=actual_right),
                expected_left.add(1, x2=expected_right),
            ),
        )
        for case, actual_legacy, expected_legacy in legacy_calls:
            self.assert_matches(
                actual_legacy,
                expected_legacy,
                case=("legacy positional default alpha", case),
            )

        actual_offset = actual_left[1]
        expected_offset = expected_left[1]
        for scalar in (True, -2, 2.5, np.bool_(True), np.int64(3), np.float32(-0.0)):
            self.assert_matches(
                actual_offset.add(scalar),
                expected_offset.add(scalar),
                case=("offset scalar", type(scalar).__name__, scalar),
            )
            self.assert_matches(
                actual_offset.add(other=scalar),
                expected_offset.add(other=scalar),
                case=("keyword scalar", type(scalar).__name__, scalar),
            )

        actual_empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        expected_empty = reference_torch.zeros((2, 0, 3)).transpose(0, 2)
        self.assert_matches(
            actual_empty.add(torch.ones((1, 1, 2))),
            expected_empty.add(reference_torch.ones((1, 1, 2))),
            case="strided broadcast empty",
        )

        special_bits = np.asarray(
            (0x0000_0000, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC1_2345),
            dtype=np.uint32,
        )
        values = memoryview(special_bits.view(np.float32))
        self.assert_matches(
            torch.tensor(values).add(torch.zeros((5,))),
            reference_torch.tensor(values).add(reference_torch.zeros((5,))),
            case="signed zero and non-finites",
        )

    def test_autograd_and_no_grad_match_pytorch_2_13(self):
        actual_left = torch.tensor([[2.0, 3.0]], requires_grad=True)
        expected_left = reference_torch.tensor([[2.0, 3.0]], requires_grad=True)
        actual_right = torch.tensor([[5.0], [7.0], [11.0]], requires_grad=True)
        expected_right = reference_torch.tensor([[5.0], [7.0], [11.0]], requires_grad=True)

        actual_output = actual_left.transpose(0, 1).add(actual_right.transpose(0, 1))
        expected_output = expected_left.transpose(0, 1).add(
            expected_right.transpose(0, 1)
        )
        self.assert_matches(actual_output, expected_output, case="views")
        actual_output.sum().backward()
        expected_output.sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_left.grad), expected_left.grad.numpy()
        )
        np.testing.assert_array_equal(
            np.asarray(actual_right.grad), expected_right.grad.numpy()
        )

        actual_shared = torch.tensor([2.0, -3.0], requires_grad=True)
        expected_shared = reference_torch.tensor([2.0, -3.0], requires_grad=True)
        actual_shared.add(actual_shared).sum().backward()
        expected_shared.add(expected_shared).sum().backward()
        self.assert_matches(
            actual_shared.grad, expected_shared.grad, case="shared operand gradient"
        )

        actual_scalar = torch.tensor([2.0, -3.0], requires_grad=True)
        expected_scalar = reference_torch.tensor([2.0, -3.0], requires_grad=True)
        actual_scalar.add(4.0).sum().backward()
        expected_scalar.add(4.0).sum().backward()
        self.assert_matches(actual_scalar.grad, expected_scalar.grad, case="scalar gradient")

        actual_empty = torch.zeros((2, 0, 3), requires_grad=True)
        expected_empty = reference_torch.zeros((2, 0, 3), requires_grad=True)
        actual_empty.add(torch.ones((1, 1, 3))).sum().backward()
        expected_empty.add(reference_torch.ones((1, 1, 3))).sum().backward()
        self.assert_matches(actual_empty.grad, expected_empty.grad, case="empty gradient")

        actual_no_grad = torch.tensor([[1.0, 2.0]], requires_grad=True)
        expected_no_grad = reference_torch.tensor([[1.0, 2.0]], requires_grad=True)
        with torch.no_grad():
            actual_untracked = actual_no_grad.transpose(0, 1).add(2.0)
        with reference_torch.no_grad():
            expected_untracked = expected_no_grad.transpose(0, 1).add(2.0)
        self.assert_matches(actual_untracked, expected_untracked, case="no_grad view")

    @staticmethod
    def dispatch_observation(module):
        left = module.tensor([2.0])
        right = module.tensor([3.0])
        descriptor = inspect.getattr_static(module.Tensor, "add")
        marker = object()
        mode_observations = []

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        def legacy_keyword_other_call():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                return left.add(1, other=right)

        for call, keyword_names in (
            (lambda: left.add(right), None),
            (lambda: left.add(4.0), None),
            (legacy_keyword_other_call, ("other",)),
            (lambda: left.add(other=right), ("other",)),
            (lambda: left.add(x2=right), ("x2",)),
            (lambda: left.add(right, alpha=True), ("alpha",)),
        ):
            mode = RecordingMode()
            with mode:
                result = call()
            func, dispatch_types, args, kwargs = mode.calls[0]
            mode_observations.append(
                (
                    result is marker,
                    func is descriptor,
                    dispatch_types == (),
                    len(args),
                    kwargs is None,
                    None if kwargs is None else tuple(kwargs),
                    keyword_names,
                )
            )

        override_observations = []

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        for call, keyword in (
            (lambda value: left.add(value), None),
            (lambda value: left.add(right, alpha=value), "alpha"),
        ):
            value = Override()
            Override.calls.clear()
            result = call(value)
            func, dispatch_types, args, kwargs = Override.calls[0]
            override_observations.append(
                (
                    result is marker,
                    func is descriptor,
                    tuple(item.__name__ for item in dispatch_types),
                    len(args),
                    kwargs is None,
                    None if kwargs is None else tuple(kwargs),
                    keyword is not None
                    and kwargs is not None
                    and kwargs[keyword] is value,
                )
            )

        value = Override()
        Override.calls.clear()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = left.add(value, right)
        func, dispatch_types, args, kwargs = Override.calls[0]
        override_observations.append(
            (
                result is marker,
                func is descriptor,
                tuple(item.__name__ for item in dispatch_types),
                len(args),
                kwargs is None,
                None if kwargs is None else tuple(kwargs),
                args[1] is value,
            )
        )

        invalid_observations = []
        for call in (
            lambda: left.add([]),
            lambda: left.add(right, out=right),
        ):
            invalid_mode = RecordingMode()
            try:
                with invalid_mode:
                    call()
            except Exception as error:
                invalid_observations.append(
                    (type(error).__name__, str(error), len(invalid_mode.calls))
                )

        return mode_observations, override_observations, invalid_observations

    def test_torch_function_mode_and_operand_dispatch_match_pytorch_2_13(self):
        self.assertEqual(
            self.dispatch_observation(torch),
            self.dispatch_observation(reference_torch),
        )

    @staticmethod
    def callable_contract(module):
        descriptor = inspect.getattr_static(module.Tensor, "add")
        tensor = module.tensor([1.0])
        bound = tensor.add
        try:
            inspect.signature(descriptor)
        except Exception as error:
            signature_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
            )
        else:
            signature_error = None
        return {
            "descriptor_type": type(descriptor).__name__,
            "descriptor_is_method_descriptor": type(descriptor)
            is types.MethodDescriptorType,
            "bound_type": type(bound).__name__,
            "bound_is_builtin_method": type(bound) is types.BuiltinMethodType,
            "name": descriptor.__name__,
            "qualname": descriptor.__qualname__,
            "bound_name": bound.__name__,
            "bound_qualname": bound.__qualname__,
            "descriptor_doc": descriptor.__doc__,
            "bound_doc": bound.__doc__,
            "text_signature": descriptor.__text_signature__,
            "bound_text_signature": bound.__text_signature__,
            "signature_error": signature_error,
            "objclass_name": descriptor.__objclass__.__name__,
            "objclass_module": descriptor.__objclass__.__module__,
            "has_descriptor_module": hasattr(descriptor, "__module__"),
            "bound_module": bound.__module__,
            "descriptor_copy_identity": copy.copy(descriptor) is descriptor,
            "descriptor_deepcopy_identity": copy.deepcopy(descriptor) is descriptor,
            "bound_copy_identity": copy.copy(bound) is bound,
            "bound_deepcopy_identity": copy.deepcopy(bound) is bound,
            "descriptor_pickle_identity": tuple(
                pickle.loads(pickle.dumps(descriptor, protocol)) is descriptor
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_documentation_copy_and_pickle_match_pytorch_2_13(self):
        self.assertEqual(self.callable_contract(torch), self.callable_contract(reference_torch))

    def test_common_binding_and_scalar_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        actual_other = torch.tensor([3.0])
        expected_other = reference_torch.tensor([3.0])
        cases = (
            (lambda: actual.add(), lambda: expected.add()),
            (lambda: actual.add([]), lambda: expected.add([])),
            (lambda: actual.add(None), lambda: expected.add(None)),
            (lambda: actual.add(actual_other, out=actual), lambda: expected.add(expected_other, out=expected)),
            (lambda: actual.add(actual_other, dtype=torch.float32), lambda: expected.add(expected_other, dtype=reference_torch.float32)),
            (lambda: actual.add(actual_other, device=torch.device("cpu")), lambda: expected.add(expected_other, device=reference_torch.device("cpu"))),
            (lambda: actual.add(actual_other, alpha=True), lambda: expected.add(expected_other, alpha=True)),
            (lambda: actual.add(np.uint64(2**63)), lambda: expected.add(np.uint64(2**63))),
            (lambda: actual.add(2**64), lambda: expected.add(2**64)),
            (lambda: actual.add(-(2**63) - 1), lambda: expected.add(-(2**63) - 1)),
        )
        for index, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=index):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
