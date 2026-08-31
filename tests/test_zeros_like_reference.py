import copy
import importlib
import inspect
import pickle
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class ZerosLikeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "zeros_like differentials require pinned PyTorch 2.13.0"
            )

    def tensor_observation(self, module, tensor):
        value_tensor = tensor.detach() if tensor.requires_grad else tensor
        return (
            tuple(tensor.shape),
            tensor.stride(),
            np.asarray(value_tensor).reshape(-1).view(np.uint32).tolist(),
            str(tensor.dtype),
            tensor.dtype is module.float32,
            str(tensor.device),
            str(tensor.layout),
            tensor.requires_grad,
            tensor.is_leaf,
            tensor.storage_offset(),
        )

    def capture_error(self, call):
        with self.assertRaises(Exception) as raised:
            call()
        return type(raised.exception), str(raised.exception)

    def make_inputs(self, module):
        return (
            ("scalar", module.tensor(-7.0, dtype=module.float32)),
            ("empty", module.zeros((2, 0, 3), dtype=module.float32)),
            (
                "multidimensional",
                module.tensor(
                    [[1.0, -2.0, 3.5], [4.0, 5.0, -6.0]],
                    dtype=module.float32,
                ),
            ),
            (
                "requires grad input",
                module.ones((2, 3), dtype=module.float32, requires_grad=True)
                * 2.0,
            ),
        )

    def option_cases(self, module):
        return (
            {},
            {"dtype": None},
            {"dtype": module.float32},
            {"layout": None},
            {"layout": module.strided},
            {"device": None},
            {"device": "cpu"},
            {"device": "cpu:0"},
            {"device": module.device("cpu")},
            {"memory_format": None},
            {"memory_format": module.preserve_format},
            {"memory_format": module.contiguous_format},
            {
                "dtype": module.float32,
                "layout": module.strided,
                "device": module.device("cpu"),
                "requires_grad": True,
                "memory_format": module.preserve_format,
            },
        )

    def test_supported_outputs_match_pytorch_2_13(self):
        actual_inputs = self.make_inputs(torch)
        expected_inputs = self.make_inputs(reference_torch)
        for (case, actual_source), (expected_case, expected_source) in zip(
            actual_inputs, expected_inputs, strict=True
        ):
            self.assertEqual(case, expected_case)
            for actual_options, expected_options in zip(
                self.option_cases(torch),
                self.option_cases(reference_torch),
                strict=True,
            ):
                with self.subTest(case=case, options=actual_options):
                    actual = torch.zeros_like(actual_source, **actual_options)
                    expected = reference_torch.zeros_like(
                        expected_source, **expected_options
                    )
                    self.assertEqual(
                        self.tensor_observation(torch, actual),
                        self.tensor_observation(reference_torch, expected),
                    )
                    self.assertEqual(
                        actual.is_set_to(actual_source),
                        expected.is_set_to(expected_source),
                    )
                    self.assertFalse(actual.is_set_to(actual_source))
                    if actual_source.numel() != 0:
                        self.assertNotEqual(actual.data_ptr(), actual_source.data_ptr())
                        self.assertNotEqual(
                            expected.data_ptr(), expected_source.data_ptr()
                        )

    def test_no_grad_factory_metadata_matches_pytorch_2_13(self):
        actual_source = torch.ones((2, 3), requires_grad=True) * 3.0
        expected_source = reference_torch.ones(
            (2, 3), requires_grad=True
        ) * 3.0

        with torch.no_grad():
            actual_default = torch.zeros_like(actual_source)
            actual_requested = torch.zeros_like(
                actual_source, requires_grad=True
            )
        with reference_torch.no_grad():
            expected_default = reference_torch.zeros_like(expected_source)
            expected_requested = reference_torch.zeros_like(
                expected_source, requires_grad=True
            )

        self.assertEqual(
            self.tensor_observation(torch, actual_default),
            self.tensor_observation(reference_torch, expected_default),
        )
        self.assertEqual(
            self.tensor_observation(torch, actual_requested),
            self.tensor_observation(reference_torch, expected_requested),
        )

    def test_binding_errors_match_pytorch_2_13(self):
        actual = torch.ones((2, 3), dtype=torch.float32)
        expected = reference_torch.ones((2, 3), dtype=reference_torch.float32)
        cases = (
            ("missing input", lambda module, tensor: module.zeros_like()),
            ("two positional", lambda module, tensor: module.zeros_like(tensor, tensor)),
            (
                "duplicate input",
                lambda module, tensor: module.zeros_like(tensor, input=tensor),
            ),
            ("out none", lambda module, tensor: module.zeros_like(tensor, out=None)),
            ("out tensor", lambda module, tensor: module.zeros_like(tensor, out=tensor)),
            (
                "invalid dtype",
                lambda module, tensor: module.zeros_like(tensor, dtype=object()),
            ),
            (
                "invalid layout",
                lambda module, tensor: module.zeros_like(tensor, layout=object()),
            ),
            (
                "invalid device",
                lambda module, tensor: module.zeros_like(tensor, device=object()),
            ),
            (
                "invalid requires_grad",
                lambda module, tensor: module.zeros_like(tensor, requires_grad=1),
            ),
            (
                "invalid memory format",
                lambda module, tensor: module.zeros_like(
                    tensor, memory_format=object()
                ),
            ),
            (
                "invalid input",
                lambda module, tensor: module.zeros_like([1.0]),
            ),
            (
                "unexpected keyword",
                lambda module, tensor: module.zeros_like(tensor, unexpected=True),
            ),
        )
        for case, call in cases:
            with self.subTest(case=case):
                actual_type, actual_message = self.capture_error(
                    lambda: call(torch, actual)
                )
                expected_type, expected_message = self.capture_error(
                    lambda: call(reference_torch, expected)
                )
                self.assertIs(actual_type, expected_type)
                self.assertEqual(actual_message, expected_message)

    def test_callable_metadata_imports_copy_pickle_and_reload_match_pytorch_2_13(self):
        package = importlib.import_module("torch_rs")
        native = package._C
        actual = package.zeros_like
        expected = reference_torch.zeros_like

        for function in (actual, expected):
            self.assertIs(type(function), types.BuiltinFunctionType)
            self.assertEqual(function.__name__, "zeros_like")
            self.assertEqual(
                function.__qualname__, "_VariableFunctionsClass.zeros_like"
            )
            self.assertEqual(function.__module__, "torch")
            self.assertIsNone(function.__text_signature__)
            with self.assertRaises(ValueError):
                inspect.signature(function)
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)

        actual_owner = actual.__reduce__()[1][0]
        expected_owner = expected.__reduce__()[1][0]
        self.assertEqual(actual_owner.__name__, expected_owner.__name__)
        self.assertEqual(actual_owner.__qualname__, expected_owner.__qualname__)
        self.assertIs(actual_owner, package._C._VariableFunctionsClass)
        self.assertIs(expected_owner, reference_torch._C._VariableFunctionsClass)
        self.assertIs(actual_owner.zeros_like, actual)
        self.assertIs(expected_owner.zeros_like, expected)
        self.assertIs(native.zeros_like, actual)
        self.assertEqual(package.__all__.count("zeros_like"), 1)
        self.assertEqual(
            package.__all__.count("zeros_like"),
            reference_torch.__all__.count("zeros_like"),
        )
        self.assertNotIn("_VariableFunctionsClass", package.__all__)
        self.assertFalse(hasattr(package, "_VariableFunctionsClass"))
        for module, function in ((package, actual), (reference_torch, expected)):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertIs(namespace["zeros_like"], function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(module=module.__name__, protocol=protocol):
                    self.assertIs(
                        pickle.loads(pickle.dumps(function, protocol=protocol)),
                        function,
                    )

        self.assertIs(importlib.reload(native), native)
        self.assertIs(native.zeros_like, actual)
        self.assertIs(importlib.reload(package), package)
        self.assertIs(package.zeros_like, actual)


if __name__ == "__main__":
    unittest.main()
