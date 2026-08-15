import inspect
import pickle
import re
import sys
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class GetDeviceReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "get_device differentials require pinned PyTorch 2.13.0"
            )

    def make_cases(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        tracked.sum().backward()
        offset_view = module.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ]
        ).transpose(0, 1)[1]
        extreme_empty = (
            module.zeros((0,))
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2)
        )
        return (
            module.tensor(3.5),
            module.zeros((2, 0, 3)),
            offset_view,
            extreme_empty,
            leaf,
            tracked,
            leaf.grad,
        )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(
            type(actual_raised.exception).__name__,
            type(expected_raised.exception).__name__,
        )
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def signature_outcome(self, callable_object):
        try:
            return "signature", inspect.signature(callable_object)
        except Exception as error:
            return "error", type(error)

    def top_level_callable_contract(self, module):
        function = module.get_device
        owner = function.__reduce__()[1][0]
        wildcard_namespace = {}
        exec(f"from {module.__name__} import *", wildcard_namespace)
        try:
            inspect.signature(function)
        except Exception as error:
            signature_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
            )
        else:
            signature_error = None
        return {
            "type": type(function).__name__,
            "is_builtin": type(function) is types.BuiltinFunctionType,
            "name": function.__name__,
            "qualname": function.__qualname__,
            "module": function.__module__,
            "owner_name": owner.__name__,
            "owner_qualname": owner.__qualname__,
            "owner_module": owner.__module__.replace("torch_rs._C", "torch._C"),
            "owner_path_identity": owner is module._C._VariableFunctionsClass,
            "owner_callable_identity": owner.get_device is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("get_device"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["get_device"] is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_cpu_results_and_device_indices_match_pytorch_2_13(self):
        actual_cases = self.make_cases(torch)
        expected_cases = self.make_cases(reference_torch)
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case, shape=actual.shape):
                actual_results = (
                    actual.get_device(),
                    torch.get_device(actual),
                    torch.get_device(input=actual),
                    torch.get_device(x=actual),
                    torch.get_device(a=actual),
                )
                expected_results = (
                    expected.get_device(),
                    reference_torch.get_device(expected),
                    reference_torch.get_device(input=expected),
                    reference_torch.get_device(x=expected),
                    reference_torch.get_device(a=expected),
                )
                self.assertEqual(actual_results, expected_results)
                self.assertTrue(
                    all(type(result) is int for result in actual_results)
                )
                self.assertEqual(actual.device.index, expected.device.index)
                self.assertEqual(
                    actual.get_device(),
                    -1 if actual.device.index is None else actual.device.index,
                )

    def test_top_level_queries_preserve_autograd_behavior(self):
        actual_leaf = torch.tensor([1.0, 2.0], requires_grad=True)
        expected_leaf = reference_torch.tensor([1.0, 2.0], requires_grad=True)
        actual_tracked = actual_leaf * 3.0
        expected_tracked = expected_leaf * 3.0

        self.assertEqual(
            (
                torch.get_device(actual_tracked),
                torch.get_device(input=actual_tracked),
                torch.get_device(x=actual_tracked),
                torch.get_device(a=actual_tracked),
            ),
            (
                reference_torch.get_device(expected_tracked),
                reference_torch.get_device(input=expected_tracked),
                reference_torch.get_device(x=expected_tracked),
                reference_torch.get_device(a=expected_tracked),
            ),
        )
        actual_tracked.sum().backward()
        expected_tracked.sum().backward()
        self.assertEqual(actual_leaf.grad.tolist(), expected_leaf.grad.tolist())

    def test_descriptor_and_argument_contract_matches_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        actual_descriptor = inspect.getattr_static(torch.Tensor, "get_device")
        expected_descriptor = inspect.getattr_static(
            reference_torch.Tensor, "get_device"
        )
        actual_bound = actual.get_device
        expected_bound = expected.get_device

        for actual_callable, expected_callable, expected_type in (
            (actual_descriptor, expected_descriptor, types.MethodDescriptorType),
            (actual_bound, expected_bound, types.BuiltinMethodType),
        ):
            self.assertIs(type(actual_callable), expected_type)
            self.assertIs(type(expected_callable), expected_type)
            self.assertEqual(actual_callable.__name__, expected_callable.__name__)
            self.assertEqual(
                actual_callable.__text_signature__,
                expected_callable.__text_signature__,
            )
            self.assertEqual(actual_callable.__doc__, expected_callable.__doc__)
            self.assertEqual(
                self.signature_outcome(actual_callable),
                self.signature_outcome(expected_callable),
            )

        self.assertEqual(
            actual_descriptor.__objclass__.__name__,
            expected_descriptor.__objclass__.__name__,
        )
        self.assertEqual(
            actual_descriptor.__objclass__.__module__,
            expected_descriptor.__objclass__.__module__,
        )
        self.assertEqual(actual_descriptor(actual), expected_descriptor(expected))
        self.assertEqual(
            self.top_level_callable_contract(torch),
            self.top_level_callable_contract(reference_torch),
        )

        call_pairs = (
            (lambda: actual.get_device(1), lambda: expected.get_device(1)),
            (lambda: actual_bound(1, 2), lambda: expected_bound(1, 2)),
            (
                lambda: actual_descriptor(actual, 1),
                lambda: expected_descriptor(expected, 1),
            ),
            (
                lambda: actual.get_device(dim=0),
                lambda: expected.get_device(dim=0),
            ),
            (
                lambda: actual_descriptor(actual, unexpected=True),
                lambda: expected_descriptor(expected, unexpected=True),
            ),
            (lambda: actual_descriptor(), lambda: expected_descriptor()),
            (lambda: actual_descriptor(1), lambda: expected_descriptor(1)),
        )
        for case, (actual_call, expected_call) in enumerate(call_pairs):
            with self.subTest(invalid_call=case):
                self.assert_error_matches(actual_call, expected_call)

        top_level_call_pairs = (
            (lambda: torch.get_device(), lambda: reference_torch.get_device()),
            (
                lambda: torch.get_device(actual, actual),
                lambda: reference_torch.get_device(expected, expected),
            ),
            (
                lambda: torch.get_device(actual, input=actual),
                lambda: reference_torch.get_device(expected, input=expected),
            ),
            (
                lambda: torch.get_device(actual, x=actual),
                lambda: reference_torch.get_device(expected, x=expected),
            ),
            (
                lambda: torch.get_device(actual, a=actual),
                lambda: reference_torch.get_device(expected, a=expected),
            ),
            (
                lambda: torch.get_device(input=actual, a=actual),
                lambda: reference_torch.get_device(input=expected, a=expected),
            ),
            (
                lambda: torch.get_device(foo=actual),
                lambda: reference_torch.get_device(foo=expected),
            ),
            (
                lambda: torch.get_device(actual, extra=True),
                lambda: reference_torch.get_device(expected, extra=True),
            ),
            (lambda: torch.get_device(1), lambda: reference_torch.get_device(1)),
            (
                lambda: torch.get_device(input=[]),
                lambda: reference_torch.get_device(input=[]),
            ),
            (
                lambda: torch.get_device(x=None),
                lambda: reference_torch.get_device(x=None),
            ),
            (
                lambda: torch.get_device(a=1),
                lambda: reference_torch.get_device(a=1),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(top_level_call_pairs):
            with self.subTest(invalid_top_level_call=case):
                self.assert_error_matches(actual_call, expected_call)

    @unittest.skipUnless(
        reference_torch is not None and reference_torch.cuda.is_available(),
        "PyTorch CUDA is unavailable",
    )
    def test_reference_cuda_ordinals_match_device_metadata(self):
        base = reference_torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], device="cuda:0"
        )
        for case, tensor in (
            ("contiguous", base),
            ("strided view", base.transpose(0, 1)),
            ("scalar view", base[0, 0]),
            ("empty", reference_torch.empty((2, 0, 3), device="cuda:0")),
        ):
            with self.subTest(case=case):
                results = (
                    tensor.get_device(),
                    reference_torch.get_device(tensor),
                    reference_torch.get_device(input=tensor),
                    reference_torch.get_device(x=tensor),
                    reference_torch.get_device(a=tensor),
                )
                self.assertEqual(results, (0, 0, 0, 0, 0))
                self.assertEqual(tensor.device.index, 0)


if __name__ == "__main__":
    unittest.main()
