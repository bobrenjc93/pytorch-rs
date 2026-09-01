import copy
import importlib
import inspect
import json
import os
import pickle
import pickletools
import re
import subprocess
import sys
import types
import typing
import unittest
from unittest import mock

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


SUPPORTED_CUDA_PROBES = {"device_count", "is_available", "is_initialized"}


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CudaProbeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "cuda probe differentials require pinned PyTorch 2.13.0"
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

    def test_signature_annotations_and_exports_match_pytorch_2_13(self):
        actual_module = importlib.import_module("torch_rs.cuda")
        expected_module = importlib.import_module("torch.cuda")

        self.assertIs(torch.cuda, actual_module)
        self.assertIs(reference_torch.cuda, expected_module)
        self.assertEqual(
            actual_module.__all__,
            [name for name in expected_module.__all__ if name in SUPPORTED_CUDA_PROBES],
        )
        self.assertEqual(
            torch.__all__.count("cuda"),
            reference_torch.__all__.count("cuda"),
        )

        for name in actual_module.__all__:
            with self.subTest(name=name):
                actual = getattr(actual_module, name)
                expected = getattr(expected_module, name)
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
                self.assertEqual(actual.__defaults__, expected.__defaults__)
                self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
                self.assertEqual(actual.__dict__, expected.__dict__)
                self.assertEqual(
                    hasattr(actual, "__text_signature__"),
                    hasattr(expected, "__text_signature__"),
                )

    def test_argument_errors_match_pytorch_2_13(self):
        cases = (
            ("is_available", "enabled"),
            ("device_count", "device"),
            ("is_initialized", "enabled"),
        )
        for name, unexpected_keyword in cases:
            actual = getattr(torch.cuda, name)
            expected = getattr(reference_torch.cuda, name)
            with self.subTest(name=name, case="one positional"):
                self.assert_error_matches(
                    lambda actual=actual: actual(None),
                    lambda expected=expected: expected(None),
                )
            with self.subTest(name=name, case="two positional"):
                self.assert_error_matches(
                    lambda actual=actual: actual(None, None),
                    lambda expected=expected: expected(None, None),
                )
            with self.subTest(name=name, case="unexpected keyword"):
                self.assert_error_matches(
                    lambda actual=actual, keyword=unexpected_keyword: actual(
                        **{keyword: True}
                    ),
                    lambda expected=expected, keyword=unexpected_keyword: expected(
                        **{keyword: True}
                    ),
                )
            with self.subTest(name=name, case="mixed"):
                self.assert_error_matches(
                    lambda actual=actual, keyword=unexpected_keyword: actual(
                        None, **{keyword: True}
                    ),
                    lambda expected=expected, keyword=unexpected_keyword: expected(
                        None, **{keyword: True}
                    ),
                )

    def test_import_wildcard_copy_and_pickle_match_supported_scope(self):
        actual_module = importlib.import_module("torch_rs.cuda")
        expected_module = importlib.import_module("torch.cuda")

        for package_name, module in (
            ("torch_rs", actual_module),
            ("torch", expected_module),
        ):
            package_import = {}
            direct_import = {}
            module_wildcard = {}
            top_level_wildcard = {}
            exec(f"from {package_name} import cuda", package_import)
            exec(
                f"from {package_name}.cuda import "
                "device_count, is_available, is_initialized",
                direct_import,
            )
            exec(f"from {package_name}.cuda import *", module_wildcard)
            exec(f"from {package_name} import *", top_level_wildcard)
            self.assertIs(package_import["cuda"], module)
            for name in SUPPORTED_CUDA_PROBES:
                self.assertIs(direct_import[name], getattr(module, name))
                self.assertIs(module_wildcard[name], getattr(module, name))
                self.assertNotIn(name, top_level_wildcard)
            self.assertNotIn("cuda", top_level_wildcard)

        for name in SUPPORTED_CUDA_PROBES:
            actual = getattr(actual_module, name)
            expected = getattr(expected_module, name)
            with self.subTest(name=name):
                self.assertIs(copy.copy(actual), actual)
                self.assertIs(copy.copy(expected), expected)
                self.assertIs(copy.deepcopy(actual), actual)
                self.assertIs(copy.deepcopy(expected), expected)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    with self.subTest(protocol=protocol):
                        self.assertIs(
                            pickle.loads(pickle.dumps(actual, protocol)),
                            actual,
                        )
                        self.assertIs(
                            pickle.loads(pickle.dumps(expected, protocol)),
                            expected,
                        )
                        self.assertEqual(
                            self.pickle_shape(actual, protocol),
                            self.pickle_shape(expected, protocol),
                        )

    def reload_contract(self, root):
        module = root.cuda
        old_functions = {
            name: getattr(module, name)
            for name in sorted(SUPPORTED_CUDA_PROBES)
        }
        namespace = module.__dict__

        reloaded = importlib.reload(module)
        new_functions = {
            name: getattr(module, name)
            for name in sorted(SUPPORTED_CUDA_PROBES)
        }

        stale_pickle_errors = []
        for name, old_function in old_functions.items():
            try:
                pickle.dumps(old_function)
            except Exception as error:
                stale_pickle_errors.append(
                    (
                        name,
                        type(error).__name__,
                        re.sub(r"0x[0-9a-fA-F]+", "0x...", str(error)).replace(
                            "torch_rs", "torch"
                        ),
                    )
                )
            else:
                self.fail(f"a stale torch.cuda.{name} probe remained pickleable")

        return (
            reloaded is module,
            module.__dict__ is namespace,
            root.cuda is module,
            sys.modules[module.__name__] is module,
            tuple(
                old_functions[name] is not new_functions[name]
                for name in sorted(SUPPORTED_CUDA_PROBES)
            ),
            tuple(
                copy.copy(new_functions[name]) is new_functions[name]
                for name in sorted(SUPPORTED_CUDA_PROBES)
            ),
            tuple(
                copy.deepcopy(new_functions[name]) is new_functions[name]
                for name in sorted(SUPPORTED_CUDA_PROBES)
            ),
            tuple(
                pickle.loads(pickle.dumps(new_functions[name])) is new_functions[name]
                for name in sorted(SUPPORTED_CUDA_PROBES)
            ),
            tuple(stale_pickle_errors),
        )

    def test_reload_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_contract(torch),
            self.reload_contract(reference_torch),
        )
        for name in SUPPORTED_CUDA_PROBES:
            actual = getattr(torch.cuda, name)
            expected = getattr(reference_torch.cuda, name)
            with self.subTest(name=name):
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    with self.subTest(protocol=protocol):
                        self.assertEqual(
                            self.pickle_shape(actual, protocol),
                            self.pickle_shape(expected, protocol),
                        )

    def test_cpu_build_probe_values_stay_fixed_when_reference_runtime_is_available(self):
        for environment in (
            {},
            {"CUDA_VISIBLE_DEVICES": ""},
            {"CUDA_VISIBLE_DEVICES": "0"},
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "NVIDIA_VISIBLE_DEVICES": "all",
                "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
            },
        ):
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    self.assertIs(torch.cuda.is_available(), False)
                    self.assertEqual(torch.cuda.device_count(), 0)
                    self.assertIs(torch.cuda.is_initialized(), False)

        self.assertTrue(hasattr(reference_torch.cuda, "is_initialized"))

    def test_fresh_subprocess_initialization_behavior_matches_supported_boundary(self):
        script = r'''
import json
import os
import pickle

os.environ.update(
    CUDA_VISIBLE_DEVICES="0",
    NVIDIA_VISIBLE_DEVICES="all",
    PYTORCH_NVML_BASED_CUDA_CHECK="1",
)

import torch as reference_torch
import torch_rs as actual_torch
from torch.cuda import device_count as expected_device_count
from torch.cuda import is_available as expected_is_available
from torch.cuda import is_initialized as expected_is_initialized
from torch_rs import cuda as actual_cuda
from torch_rs.cuda import device_count, is_available, is_initialized

module_wildcard = {}
exec("from torch_rs.cuda import *", module_wildcard)

payload = {
    "actual_values": [
        actual_cuda.device_count(),
        actual_cuda.is_available(),
        actual_cuda.is_initialized(),
    ],
    "actual_types": [
        type(actual_cuda.device_count()).__name__,
        type(actual_cuda.is_available()).__name__,
        type(actual_cuda.is_initialized()).__name__,
    ],
    "actual_imports": [
        actual_torch.cuda is actual_cuda,
        actual_cuda.device_count is device_count,
        actual_cuda.is_available is is_available,
        actual_cuda.is_initialized is is_initialized,
        sorted(name for name in module_wildcard if not name.startswith("__")),
    ],
    "actual_all": actual_cuda.__all__,
    "actual_code_names": [
        actual_cuda.device_count.__code__.co_names,
        actual_cuda.is_available.__code__.co_names,
        actual_cuda.is_initialized.__code__.co_names,
    ],
    "actual_pickle": [
        pickle.loads(pickle.dumps(actual_cuda.device_count)) is actual_cuda.device_count,
        pickle.loads(pickle.dumps(actual_cuda.is_available)) is actual_cuda.is_available,
        pickle.loads(pickle.dumps(actual_cuda.is_initialized)) is actual_cuda.is_initialized,
    ],
    "reference_initial_is_initialized": reference_torch.cuda.is_initialized(),
    "reference_types": [
        type(expected_device_count()).__name__,
        type(expected_is_available()).__name__,
        type(expected_is_initialized()).__name__,
    ],
    "reference_after_probes_is_initialized": reference_torch.cuda.is_initialized(),
}
print(json.dumps(payload, sort_keys=True))
'''
        completed = subprocess.run(
            [sys.executable, "-W", "ignore", "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )
        payload = json.loads(completed.stdout.strip().splitlines()[-1])

        self.assertEqual(payload["actual_values"], [0, False, False])
        self.assertEqual(payload["actual_types"], ["int", "bool", "bool"])
        self.assertEqual(
            payload["actual_imports"],
            [
                True,
                True,
                True,
                True,
                ["device_count", "is_available", "is_initialized"],
            ],
        )
        self.assertEqual(
            payload["actual_all"],
            ["device_count", "is_available", "is_initialized"],
        )
        self.assertEqual(payload["actual_code_names"], [[], [], []])
        self.assertEqual(payload["actual_pickle"], [True, True, True])
        self.assertIs(payload["reference_initial_is_initialized"], False)
        self.assertEqual(payload["reference_types"], ["int", "bool", "bool"])
        self.assertIs(payload["reference_after_probes_is_initialized"], False)


if __name__ == "__main__":
    unittest.main()
