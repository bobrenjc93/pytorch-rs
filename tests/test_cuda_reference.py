import copy
import importlib
import inspect
import pickle
import pickletools
import sys
import types
import typing
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CudaProbeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("cuda probe differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def pickle_shape(self, function, protocol):
        shape = []
        for opcode, argument, _ in pickletools.genops(
            pickle.dumps(function, protocol=protocol)
        ):
            if opcode.name == "FRAME":
                argument = "<frame length>"
            elif isinstance(argument, str):
                argument = argument.replace("torch_rs", "torch")
            shape.append((opcode.name, argument))
        return shape

    def test_signature_documentation_and_identity_match_pytorch_2_13(self):
        actual_module = importlib.import_module("torch_rs.cuda")
        expected_module = importlib.import_module("torch.cuda")

        self.assertIs(torch.cuda, actual_module)
        self.assertIs(reference_torch.cuda, expected_module)
        self.assertIs(sys.modules["torch_rs.cuda"], actual_module)
        self.assertIs(sys.modules["torch.cuda"], expected_module)
        self.assertIsNot(actual_module, torch.backends.cuda)
        self.assertEqual(
            actual_module.__all__,
            [name for name in expected_module.__all__ if name in actual_module.__all__],
        )

        for name in ("device_count", "is_available"):
            with self.subTest(name=name):
                actual = getattr(actual_module, name)
                expected = getattr(expected_module, name)
                self.assertIs(type(actual), types.FunctionType)
                self.assertIs(type(expected), types.FunctionType)
                self.assertEqual(str(inspect.signature(actual)), str(inspect.signature(expected)))
                self.assertEqual(actual.__annotations__, expected.__annotations__)
                self.assertEqual(typing.get_type_hints(actual), typing.get_type_hints(expected))
                self.assertEqual(actual.__name__, expected.__name__)
                self.assertEqual(actual.__qualname__, expected.__qualname__)
                self.assertEqual(
                    actual.__module__.replace("torch_rs", "torch"),
                    expected.__module__,
                )
                self.assertIs(inspect.getmodule(actual), actual_module)
                self.assertIs(inspect.getmodule(expected), expected_module)
                self.assertEqual(actual.__doc__, expected.__doc__)
                self.assertEqual(actual.__defaults__, expected.__defaults__)
                self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
                self.assertEqual(actual.__dict__, expected.__dict__)
                self.assertEqual(
                    hasattr(actual, "__text_signature__"),
                    hasattr(expected, "__text_signature__"),
                )

    def test_imports_wildcards_copy_and_pickle_match_supported_subset(self):
        actual_module = torch.cuda
        expected_module = reference_torch.cuda

        for package_name, module in (
            ("torch_rs", actual_module),
            ("torch", expected_module),
        ):
            package_import = {}
            function_import = {}
            exec(f"from {package_name} import cuda", package_import)
            exec(
                f"from {package_name}.cuda import device_count, is_available",
                function_import,
            )
            self.assertIs(package_import["cuda"], module)
            self.assertIs(function_import["device_count"], module.device_count)
            self.assertIs(function_import["is_available"], module.is_available)

        actual_top_wildcard = {}
        expected_top_wildcard = {}
        exec("from torch_rs import *", actual_top_wildcard)
        exec("from torch import *", expected_top_wildcard)
        self.assertNotIn("cuda", actual_top_wildcard)
        self.assertNotIn("cuda", expected_top_wildcard)

        actual_cuda_wildcard = {}
        expected_cuda_wildcard = {}
        exec("from torch_rs.cuda import *", actual_cuda_wildcard)
        exec("from torch.cuda import *", expected_cuda_wildcard)
        self.assertEqual(
            {name for name in actual_cuda_wildcard if not name.startswith("__")},
            {
                name
                for name in expected_cuda_wildcard
                if name in {"device_count", "is_available"}
            },
        )

        for name in ("device_count", "is_available"):
            with self.subTest(name=name):
                actual = getattr(actual_module, name)
                expected = getattr(expected_module, name)
                self.assertIs(copy.copy(actual), actual)
                self.assertIs(copy.copy(expected), expected)
                self.assertIs(copy.deepcopy(actual), actual)
                self.assertIs(copy.deepcopy(expected), expected)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                    self.assertIs(
                        pickle.loads(pickle.dumps(expected, protocol)),
                        expected,
                    )
                    self.assertEqual(
                        self.pickle_shape(actual, protocol),
                        self.pickle_shape(expected, protocol),
                    )

    def test_argument_errors_match_pytorch_2_13(self):
        cases = (
            (
                lambda: torch.cuda.is_available(None),
                lambda: reference_torch.cuda.is_available(None),
            ),
            (
                lambda: torch.cuda.is_available(None, None),
                lambda: reference_torch.cuda.is_available(None, None),
            ),
            (
                lambda: torch.cuda.is_available(enabled=True),
                lambda: reference_torch.cuda.is_available(enabled=True),
            ),
            (
                lambda: torch.cuda.is_available(None, enabled=True),
                lambda: reference_torch.cuda.is_available(None, enabled=True),
            ),
            (
                lambda: torch.cuda.device_count(None),
                lambda: reference_torch.cuda.device_count(None),
            ),
            (
                lambda: torch.cuda.device_count(None, None),
                lambda: reference_torch.cuda.device_count(None, None),
            ),
            (
                lambda: torch.cuda.device_count(enabled=True),
                lambda: reference_torch.cuda.device_count(enabled=True),
            ),
            (
                lambda: torch.cuda.device_count(None, enabled=True),
                lambda: reference_torch.cuda.device_count(None, enabled=True),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_cpu_build_values_bound_cuda_enabled_reference(self):
        self.assertIs(type(reference_torch.cuda.is_available()), bool)
        self.assertIs(type(reference_torch.cuda.device_count()), int)
        self.assertIs(torch.cuda.is_available(), False)
        self.assertEqual(torch.cuda.device_count(), 0)
        self.assertIs(torch.backends.cuda.is_built(), False)


if __name__ == "__main__":
    unittest.main()
