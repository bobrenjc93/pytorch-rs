import array
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


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class AsTensorReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("as_tensor differentials require pinned PyTorch 2.13.0")

    def tensor_cases(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        produced = leaf * 2.0
        tracked = produced.transpose(0, 1)
        source = module.tensor(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=module.float32
        )
        strided = source.transpose(0, 1)[1]
        empty = module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1]
        gradient_leaf = module.tensor(
            [2.0, 3.0], dtype=module.float32, requires_grad=True
        )
        (gradient_leaf * 4.0).sum().backward()
        return leaf, tracked, (
            module.tensor(-0.0, dtype=module.float32),
            empty,
            strided,
            leaf,
            produced,
            tracked,
            gradient_leaf.grad,
        )

    def tensor_contract(self, tensor):
        array_source = tensor.detach() if tensor.requires_grad else tensor
        return (
            np.asarray(array_source).tolist(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.storage_offset(),
            str(tensor.dtype),
            str(tensor.device),
            str(tensor.layout),
            tensor.requires_grad,
            tensor.is_leaf,
            tensor.output_nr,
        )

    def test_exact_tensor_identity_metadata_matches_pytorch_2_13(self):
        actual_leaf, actual_tracked, actual_cases = self.tensor_cases(torch)
        expected_leaf, expected_tracked, expected_cases = self.tensor_cases(
            reference_torch
        )
        option_pairs = (
            ({}, {}),
            ({"dtype": None}, {"dtype": None}),
            ({"dtype": torch.float32}, {"dtype": reference_torch.float32}),
            ({"dtype": torch.float}, {"dtype": reference_torch.float}),
            ({"device": None}, {"device": None}),
            ({"device": "cpu"}, {"device": "cpu"}),
            (
                {"device": torch.device("cpu")},
                {"device": reference_torch.device("cpu")},
            ),
            (
                {"dtype": torch.float32, "device": "cpu"},
                {"dtype": reference_torch.float32, "device": "cpu"},
            ),
        )
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            for actual_options, expected_options in option_pairs:
                with self.subTest(case=case, options=actual_options):
                    actual_before = self.tensor_contract(actual)
                    expected_before = self.tensor_contract(expected)
                    actual_result = torch.as_tensor(actual, **actual_options)
                    expected_result = reference_torch.as_tensor(
                        expected, **expected_options
                    )
                    self.assertIs(actual_result, actual)
                    self.assertIs(expected_result, expected)
                    self.assertEqual(
                        self.tensor_contract(actual_result), actual_before
                    )
                    self.assertEqual(
                        self.tensor_contract(expected_result), expected_before
                    )

        torch.as_tensor(actual_tracked).sum().backward()
        reference_torch.as_tensor(expected_tracked).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_leaf.grad), expected_leaf.grad.numpy()
        )

    def test_indexed_cpu_tensor_copy_matches_pytorch_2_13(self):
        device_pairs = (
            ("cpu:0", "cpu:0"),
            ("cpu:1", "cpu:1"),
            (torch.device("cpu", 0), reference_torch.device("cpu", 0)),
        )
        for actual_device, expected_device in device_pairs:
            with self.subTest(device=actual_device):
                actual_leaf = torch.tensor([2.0, 3.0], requires_grad=True)
                expected_leaf = reference_torch.tensor(
                    [2.0, 3.0], dtype=reference_torch.float32, requires_grad=True
                )
                actual = torch.as_tensor(actual_leaf, device=actual_device)
                expected = reference_torch.as_tensor(
                    expected_leaf, device=expected_device
                )
                self.assertIs(actual is actual_leaf, expected is expected_leaf)
                self.assertEqual(
                    self.tensor_contract(actual),
                    self.tensor_contract(expected),
                )
                self.assertNotEqual(actual.data_ptr(), actual_leaf.data_ptr())
                self.assertNotEqual(expected.data_ptr(), expected_leaf.data_ptr())
                actual.sum().backward()
                expected.sum().backward()
                np.testing.assert_array_equal(
                    np.asarray(actual_leaf.grad), expected_leaf.grad.numpy()
                )

    def test_supported_non_tensor_inputs_match_pytorch_values_without_numpy_aliasing(self):
        cases = (
            ("scalar", 1.25),
            ("sequence", [[1, 2, 3], [4, 5, 6]]),
            ("array", array.array("i", [-7, 0, 9])),
            ("bytearray", bytearray((1, 2, 3))),
            ("numpy", np.asarray([1.0, 2.0, 3.0], dtype=np.float32)),
        )
        for case, source in cases:
            with self.subTest(case=case):
                actual = torch.as_tensor(source, dtype=torch.float32, device="cpu")
                expected = reference_torch.as_tensor(
                    source,
                    dtype=reference_torch.float32,
                    device="cpu",
                )
                self.assertEqual(
                    self.tensor_contract(actual),
                    self.tensor_contract(expected),
                )

        numpy_source = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        actual = torch.as_tensor(numpy_source, dtype=torch.float32)
        expected = reference_torch.as_tensor(numpy_source, dtype=reference_torch.float32)
        self.assertNotEqual(actual.data_ptr(), numpy_source.__array_interface__["data"][0])
        self.assertEqual(expected.data_ptr(), numpy_source.__array_interface__["data"][0])
        numpy_source[0] = 9.0
        self.assertEqual(actual.tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(expected.tolist(), [9.0, 2.0, 3.0])

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

    def test_binding_and_argument_type_errors_match_pytorch_2_13(self):
        actual_tensor = torch.tensor([1.0])
        expected_tensor = reference_torch.tensor(
            [1.0], dtype=reference_torch.float32
        )
        call_pairs = (
            (lambda: torch.as_tensor(), lambda: reference_torch.as_tensor()),
            (
                lambda: torch.as_tensor([1.0], torch.float32),
                lambda: reference_torch.as_tensor([1.0], reference_torch.float32),
            ),
            (
                lambda: torch.as_tensor([1.0], data=[2.0]),
                lambda: reference_torch.as_tensor([1.0], data=[2.0]),
            ),
            (
                lambda: torch.as_tensor([1.0], requires_grad=True),
                lambda: reference_torch.as_tensor([1.0], requires_grad=True),
            ),
            (
                lambda: torch.as_tensor([1.0], dtype=1),
                lambda: reference_torch.as_tensor([1.0], dtype=1),
            ),
            (
                lambda: torch.as_tensor(actual_tensor, device=1.5),
                lambda: reference_torch.as_tensor(expected_tensor, device=1.5),
            ),
            (
                lambda: torch.as_tensor(object(), dtype=1),
                lambda: reference_torch.as_tensor(object(), dtype=1),
            ),
            (
                lambda: torch.as_tensor(object(), device=1.5),
                lambda: reference_torch.as_tensor(object(), device=1.5),
            ),
            (
                lambda: torch.as_tensor([1.0], extra=True, dtype=1),
                lambda: reference_torch.as_tensor([1.0], extra=True, dtype=1),
            ),
        )
        for actual_call, expected_call in call_pairs:
            with self.subTest(actual_call=actual_call):
                self.assert_error_matches(actual_call, expected_call)

    def callable_contract(self, module):
        function = module.as_tensor
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
            "owner_callable_identity": owner.as_tensor is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("as_tensor"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["as_tensor"] is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_documentation_and_exports_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
