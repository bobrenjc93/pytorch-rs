import copy
import importlib
import inspect
import os
import pickle
import pickletools
import re
import sys
import threading
import types
import typing
import unittest
from unittest import mock

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CuSparseLtAvailabilityReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "backends.cusparselt.is_available differentials require "
                "pinned PyTorch 2.13.0"
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

    def test_value_and_private_native_placement_match_pytorch_2_13_shape(self):
        actual = torch.backends.cusparselt.is_available()
        expected = reference_torch.backends.cusparselt.is_available()

        self.assertIs(type(actual), bool)
        self.assertIs(type(expected), bool)
        self.assertIs(actual, torch._C._has_cusparselt)
        self.assertIs(expected, reference_torch._C._has_cusparselt)
        self.assertIs(actual, False)

        native_import = {}
        package_wildcard = {}
        native_wildcard = {}
        exec("from torch_rs._C import _has_cusparselt", native_import)
        exec("from torch_rs import *", package_wildcard)
        exec("from torch_rs._C import *", native_wildcard)
        self.assertIs(native_import["_has_cusparselt"], torch._C._has_cusparselt)
        self.assertFalse(hasattr(torch, "_has_cusparselt"))
        self.assertNotIn("_has_cusparselt", torch.__all__)
        self.assertNotIn("_has_cusparselt", package_wildcard)
        self.assertNotIn("_has_cusparselt", native_wildcard)

        for root in (torch, reference_torch):
            native_flag = root._C._has_cusparselt
            self.assertIs(type(native_flag), bool)
            self.assertFalse(hasattr(root, "_has_cusparselt"))
            self.assertNotIn("_has_cusparselt", root.__all__)
            if hasattr(root._C, "__all__"):
                self.assertNotIn("_has_cusparselt", root._C.__all__)

    def test_signature_documentation_and_identity_match_pytorch_2_13(self):
        actual_module = importlib.import_module("torch_rs.backends.cusparselt")
        expected_module = importlib.import_module("torch.backends.cusparselt")
        actual = actual_module.is_available
        expected = expected_module.is_available

        self.assertIsNone(actual_module.__doc__)
        self.assertEqual(actual_module.__doc__, expected_module.__doc__)
        self.assertEqual(
            {name for name in vars(actual_module) if not name.startswith("_")},
            {"is_available", "torch"},
        )
        self.assertEqual(
            {name for name in vars(expected_module) if not name.startswith("_")},
            {"Optional", "get_max_alg_id", "is_available", "torch", "version"},
        )
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual)),
            str(inspect.signature(expected)),
        )
        self.assertEqual(
            inspect.get_annotations(actual),
            inspect.get_annotations(expected),
        )
        self.assertEqual(
            typing.get_type_hints(actual),
            typing.get_type_hints(expected),
        )
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
        self.assertEqual(actual.__code__.co_names, expected.__code__.co_names)
        self.assertEqual(actual.__code__.co_freevars, expected.__code__.co_freevars)
        self.assertEqual(actual.__code__.co_cellvars, expected.__code__.co_cellvars)

    def test_imports_copying_and_pickling_match_pytorch_2_13_for_supported_api(self):
        actual_backends = importlib.import_module("torch_rs.backends")
        expected_backends = importlib.import_module("torch.backends")
        actual_module = importlib.import_module("torch_rs.backends.cusparselt")
        expected_module = importlib.import_module("torch.backends.cusparselt")
        actual = actual_module.is_available
        expected = expected_module.is_available

        self.assertIs(torch.backends, actual_backends)
        self.assertIs(reference_torch.backends, expected_backends)
        self.assertIs(actual_backends.cusparselt, actual_module)
        self.assertIs(expected_backends.cusparselt, expected_module)
        self.assertIs(sys.modules[actual_module.__name__], actual_module)
        self.assertIs(sys.modules[expected_module.__name__], expected_module)

        for package_name, module, function in (
            ("torch_rs", actual_module, actual),
            ("torch", expected_module, expected),
        ):
            backend_import = {}
            function_import = {}
            exec(
                f"from {package_name}.backends import cusparselt",
                backend_import,
            )
            exec(
                f"from {package_name}.backends.cusparselt import is_available",
                function_import,
            )
            self.assertIs(backend_import["cusparselt"], module)
            self.assertIs(function_import["is_available"], function)

        actual_parent_wildcard = {}
        expected_parent_wildcard = {}
        exec("from torch_rs.backends import *", actual_parent_wildcard)
        exec("from torch.backends import *", expected_parent_wildcard)
        self.assertIs(actual_parent_wildcard["cusparselt"], actual_module)
        self.assertIs(expected_parent_wildcard["cusparselt"], expected_module)

        actual_child_wildcard = {}
        exec("from torch_rs.backends.cusparselt import *", actual_child_wildcard)
        self.assertEqual(
            {name for name in actual_child_wildcard if not name.startswith("__")},
            {"is_available"},
        )
        self.assertIs(actual_child_wildcard["is_available"], actual)

        self.assertIs(copy.copy(actual), actual)
        self.assertIs(copy.copy(expected), expected)
        self.assertIs(copy.deepcopy(actual), actual)
        self.assertIs(copy.deepcopy(expected), expected)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(
                    pickle.loads(pickle.dumps(expected, protocol)),
                    expected,
                )
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

    def threaded_contract(self, root):
        function = root.backends.cusparselt.is_available
        flag = root._C._has_cusparselt
        worker_count = 16
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                barrier.wait(timeout=5)
                value = function()
                results[index] = (
                    type(value) is bool,
                    value is flag,
                    function is root.backends.cusparselt.is_available,
                )
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))

        threads = [
            threading.Thread(target=worker, args=(index,))
            for index in range(worker_count)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        return (
            any(thread.is_alive() for thread in threads),
            errors,
            results,
        )

    def test_thread_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.threaded_contract(torch),
            self.threaded_contract(reference_torch),
        )

    def reload_contract(self, root):
        parent = root.backends
        module = parent.cusparselt
        old_function = module.is_available
        namespace = module.__dict__
        reloaded = importlib.reload(module)
        new_function = module.is_available

        try:
            pickle.dumps(old_function)
        except Exception as error:
            stale_pickle_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-fA-F]+", "0x...", str(error)).replace(
                    "torch_rs", "torch"
                ),
            )
        else:
            self.fail("a stale cuSPARSELt availability query remained pickleable")

        return (
            reloaded is module,
            module.__dict__ is namespace,
            parent.cusparselt is module,
            sys.modules[module.__name__] is module,
            old_function is not new_function,
            new_function() is root._C._has_cusparselt,
            copy.copy(new_function) is new_function,
            copy.deepcopy(new_function) is new_function,
            pickle.loads(pickle.dumps(new_function)) is new_function,
            stale_pickle_error,
        )

    def test_reload_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_contract(torch),
            self.reload_contract(reference_torch),
        )
        actual = torch.backends.cusparselt.is_available
        expected = reference_torch.backends.cusparselt.is_available
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.backends.cusparselt.is_available
        expected = reference_torch.backends.cusparselt.is_available
        cases = (
            ((None,), {}),
            ((None, None), {}),
            ((), {"enabled": True}),
            ((None,), {"enabled": True}),
        )
        for args, kwargs in cases:
            with self.subTest(args=args, kwargs=kwargs):
                self.assert_error_matches(
                    lambda: actual(*args, **kwargs),
                    lambda: expected(*args, **kwargs),
                )

    def test_environment_invariance_matches_build_flag_shape(self):
        actual = torch.backends.cusparselt.is_available
        expected = reference_torch.backends.cusparselt.is_available
        environments = (
            {},
            {"CUDA_VISIBLE_DEVICES": ""},
            {"CUDA_VISIBLE_DEVICES": "0"},
            {"USE_CUSPARSELT": "1"},
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "CUSPARSELT_PATH": "/not/a/cusparselt/install",
                "NVIDIA_VISIBLE_DEVICES": "all",
                "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
                "USE_CUSPARSELT": "1",
            },
        )
        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    self.assertIs(type(actual()), type(expected()))
                    self.assertIs(actual(), False)
                    self.assertIs(actual(), torch._C._has_cusparselt)
                    self.assertIs(expected(), reference_torch._C._has_cusparselt)

    def test_cuda_visible_h100_preserves_torch_rs_cpu_build_boundary(self):
        if not reference_torch.cuda.is_available():
            self.skipTest("requires a CUDA-visible reference PyTorch runtime")

        device_name = reference_torch.cuda.get_device_name(0)
        if "H100" not in device_name:
            self.skipTest(f"requires an NVIDIA H100, found {device_name}")

        self.assertGreaterEqual(reference_torch.cuda.device_count(), 1)
        self.assertIs(reference_torch._C._has_cuda, True)
        self.assertIs(reference_torch._C._has_cusparselt, True)
        self.assertIs(reference_torch.backends.cusparselt.is_available(), True)
        self.assertIs(torch._C._has_cuda, False)
        self.assertIs(torch.backends.cuda.is_built(), False)
        self.assertIs(torch._C._has_cusparselt, False)
        self.assertIs(torch.backends.cusparselt.is_available(), False)
        self.assertFalse(hasattr(torch.backends.cusparselt, "version"))
        self.assertFalse(hasattr(torch.backends.cusparselt, "get_max_alg_id"))
        self.assertFalse(hasattr(torch._C, "_cusparselt"))
        self.assertIs(torch.cuda.is_available(), False)
        self.assertEqual(torch.cuda.device_count(), 0)

    def test_execution_version_and_algorithm_apis_remain_unsupported(self):
        actual = torch.backends.cusparselt
        expected = reference_torch.backends.cusparselt
        self.assertFalse(hasattr(actual, "version"))
        self.assertFalse(hasattr(actual, "get_max_alg_id"))
        self.assertTrue(hasattr(expected, "version"))
        self.assertTrue(hasattr(expected, "get_max_alg_id"))
        self.assertFalse(hasattr(torch._C, "_cusparselt"))


if __name__ == "__main__":
    unittest.main()
