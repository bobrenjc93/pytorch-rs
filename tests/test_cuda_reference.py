import copy
import importlib
import inspect
import json
import pickle
import pickletools
import subprocess
import sys
import types
import typing
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


SUPPORTED_CUDA_PROBES = {"device_count", "is_available", "is_initialized"}


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CudaIsInitializedReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "cuda.is_initialized differentials require pinned PyTorch 2.13.0"
            )

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
        actual_cuda = importlib.import_module("torch_rs.cuda")
        expected_cuda = importlib.import_module("torch.cuda")
        actual = actual_cuda.is_initialized
        expected = expected_cuda.is_initialized

        self.assertIs(torch.cuda, actual_cuda)
        self.assertIs(reference_torch.cuda, expected_cuda)
        self.assertIs(sys.modules["torch_rs.cuda"], actual_cuda)
        self.assertIs(sys.modules["torch.cuda"], expected_cuda)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual)), str(inspect.signature(expected))
        )
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(typing.get_type_hints(actual), typing.get_type_hints(expected))
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertIs(inspect.getmodule(actual), actual_cuda)
        self.assertIs(inspect.getmodule(expected), expected_cuda)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )

    def test_imports_wildcard_copy_and_pickle_match_supported_scope(self):
        actual_cuda = torch.cuda
        expected_cuda = reference_torch.cuda
        actual = actual_cuda.is_initialized
        expected = expected_cuda.is_initialized

        self.assertEqual(
            actual_cuda.__all__,
            [name for name in expected_cuda.__all__ if name in SUPPORTED_CUDA_PROBES],
        )
        self.assertEqual(
            {name for name in vars(actual_cuda) if not name.startswith("_")},
            SUPPORTED_CUDA_PROBES,
        )

        actual_package_import = {}
        expected_package_import = {}
        exec("from torch_rs import cuda", actual_package_import)
        exec("from torch import cuda", expected_package_import)
        self.assertIs(actual_package_import["cuda"], actual_cuda)
        self.assertIs(expected_package_import["cuda"], expected_cuda)

        actual_direct_import = {}
        expected_direct_import = {}
        exec("from torch_rs.cuda import is_initialized", actual_direct_import)
        exec("from torch.cuda import is_initialized", expected_direct_import)
        self.assertIs(actual_direct_import["is_initialized"], actual)
        self.assertIs(expected_direct_import["is_initialized"], expected)

        actual_cuda_namespace = {}
        expected_cuda_namespace = {}
        exec("from torch_rs.cuda import *", actual_cuda_namespace)
        exec("from torch.cuda import *", expected_cuda_namespace)
        self.assertEqual(
            {name for name in actual_cuda_namespace if not name.startswith("__")},
            SUPPORTED_CUDA_PROBES,
        )
        for name in SUPPORTED_CUDA_PROBES:
            with self.subTest(cuda_export=name):
                self.assertIs(actual_cuda_namespace[name], getattr(actual_cuda, name))
                self.assertIs(expected_cuda_namespace[name], getattr(expected_cuda, name))

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("cuda", namespace)
            self.assertNotIn("is_initialized", namespace)

        self.assertIs(copy.copy(actual), actual)
        self.assertIs(copy.copy(expected), expected)
        self.assertIs(copy.deepcopy(actual), actual)
        self.assertIs(copy.deepcopy(expected), expected)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(pickle.loads(pickle.dumps(expected, protocol)), expected)
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.cuda.is_initialized
        expected = reference_torch.cuda.is_initialized
        cases = (
            (lambda: actual(None), lambda: expected(None)),
            (lambda: actual(None, None), lambda: expected(None, None)),
            (lambda: actual(enabled=True), lambda: expected(enabled=True)),
            (
                lambda: actual(None, enabled=True),
                lambda: expected(None, enabled=True),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_fresh_subprocess_probe_and_reload_behavior_matches_pytorch_2_13(self):
        script = r'''
import copy
import importlib
import json
import pickle
import re

import torch
import torch_rs

if torch.__version__.split("+")[0] != "2.13.0":
    raise AssertionError(f"expected PyTorch 2.13.0, got {torch.__version__}")

supported = ["device_count", "is_available", "is_initialized"]

def describe(module):
    cuda = importlib.import_module(module.__name__ + ".cuda")
    module_wildcard = {}
    top_level_wildcard = {}
    exec(f"from {module.__name__}.cuda import *", module_wildcard)
    exec(f"from {module.__name__} import *", top_level_wildcard)

    before = cuda.is_initialized()
    available = cuda.is_available()
    count = cuda.device_count()
    after = cuda.is_initialized()

    old_function = cuda.is_initialized
    namespace = cuda.__dict__
    reloaded = importlib.reload(cuda)
    old_pickle_error = None
    try:
        pickle.dumps(old_function)
    except Exception as error:
        old_pickle_error = [
            type(error).__name__,
            re.sub(r"0x[0-9a-fA-F]+", "0x...", str(error)).replace(
                module.__name__, "torch"
            ),
        ]

    return {
        "available": available,
        "available_type": type(available).__name__,
        "count": count,
        "count_type": type(count).__name__,
        "initialized_before_probes": before,
        "initialized_after_probes": after,
        "module_identity": module.cuda is cuda and reloaded is cuda,
        "namespace_preserved": cuda.__dict__ is namespace,
        "reload_replaced_function": cuda.is_initialized is not old_function,
        "reload_value": cuda.is_initialized(),
        "copy_identity": copy.copy(cuda.is_initialized) is cuda.is_initialized,
        "pickle_identity": pickle.loads(pickle.dumps(cuda.is_initialized))
        is cuda.is_initialized,
        "old_pickle_error": old_pickle_error,
        "module_wildcard_supported": [
            name for name in supported if name in module_wildcard
        ],
        "top_level_wildcard_cuda": "cuda" in top_level_wildcard,
        "top_level_wildcard_is_initialized": "is_initialized" in top_level_wildcard,
    }

print(json.dumps({"actual": describe(torch_rs), "expected": describe(torch)}))
'''
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )
        observed = json.loads(completed.stdout)
        actual = observed["actual"]
        expected = observed["expected"]

        self.assertIs(actual["available"], False)
        self.assertEqual(actual["available_type"], "bool")
        self.assertEqual(actual["count"], 0)
        self.assertEqual(actual["count_type"], "int")
        self.assertIs(actual["initialized_before_probes"], False)
        self.assertIs(expected["initialized_before_probes"], False)
        self.assertIs(actual["initialized_after_probes"], False)
        self.assertIs(expected["initialized_after_probes"], False)
        self.assertIs(actual["reload_value"], False)
        self.assertIs(expected["reload_value"], False)
        for key in (
            "module_identity",
            "namespace_preserved",
            "reload_replaced_function",
            "copy_identity",
            "pickle_identity",
            "module_wildcard_supported",
            "top_level_wildcard_cuda",
            "top_level_wildcard_is_initialized",
            "old_pickle_error",
        ):
            with self.subTest(key=key):
                self.assertEqual(actual[key], expected[key])


if __name__ == "__main__":
    unittest.main()
