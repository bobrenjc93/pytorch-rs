import copy
import functools
import importlib
import inspect
import os
import pickle
import subprocess
import sys
import types
import typing
import unittest
from unittest import mock

import torch_rs as torch


FUNCTION_DOC = """
    Returns the module associated with a given device(e.g., torch.device('cuda'), "mtia:0", "xpu", ...).
    If no device is given, return the module for the current accelerator or CPU if none is present.
    """


class GetDeviceModuleTests(unittest.TestCase):
    def setUp(self):
        torch.get_device_module.cache_clear()
        self.addCleanup(torch.get_device_module.cache_clear)

    def cache_shape(self):
        info = torch.get_device_module.cache_info()
        return info.hits, info.misses, info.maxsize, info.currsize

    def test_default_strings_and_descriptors_return_the_canonical_cpu_module(self):
        cpu = importlib.import_module("torch_rs.cpu")
        copied_device = copy.copy(torch.device("cpu:7"))
        pickled_device = pickle.loads(pickle.dumps(torch.device("cpu:127")))
        calls = (
            lambda: torch.get_device_module(),
            lambda: torch.get_device_module(None),
            lambda: torch.get_device_module(device=None),
            lambda: torch.get_device_module("cpu"),
            lambda: torch.get_device_module("cpu:0"),
            lambda: torch.get_device_module("cpu:128"),
            lambda: torch.get_device_module(torch.device("cpu")),
            lambda: torch.get_device_module(torch.device("cpu", 0)),
            lambda: torch.get_device_module(torch.device("cpu:255")),
            lambda: torch.get_device_module(copied_device),
            lambda: torch.get_device_module(pickled_device),
        )

        self.assertIs(torch.cpu, cpu)
        self.assertIs(sys.modules["torch_rs.cpu"], cpu)
        for call in calls:
            with self.subTest(call=call):
                self.assertIs(call(), cpu)

    def test_cache_keys_hits_misses_and_clear_match_functools_cache(self):
        function = torch.get_device_module
        cpu = torch.cpu
        expected_wrapper_type = type(functools.cache(lambda: None))

        self.assertIs(type(function), expected_wrapper_type)
        self.assertEqual(function.cache_parameters(), {"maxsize": None, "typed": False})
        self.assertEqual(self.cache_shape(), (0, 0, None, 0))

        calls_and_cache_shapes = (
            (lambda: function(), (0, 1, None, 1)),
            (lambda: function(), (1, 1, None, 1)),
            (lambda: function(None), (1, 2, None, 2)),
            (lambda: function(None), (2, 2, None, 2)),
            (lambda: function(device=None), (2, 3, None, 3)),
            (lambda: function("cpu"), (2, 4, None, 4)),
            (lambda: function("cpu"), (3, 4, None, 4)),
            (lambda: function(device="cpu"), (3, 5, None, 5)),
            (lambda: function("cpu:0"), (3, 6, None, 6)),
            (lambda: function(torch.device("cpu")), (3, 7, None, 7)),
            (lambda: function(torch.device("cpu")), (4, 7, None, 7)),
            (lambda: function(torch.device("cpu:0")), (4, 8, None, 8)),
        )
        for call, expected_cache_shape in calls_and_cache_shapes:
            with self.subTest(expected_cache_shape=expected_cache_shape):
                self.assertIs(call(), cpu)
                self.assertEqual(self.cache_shape(), expected_cache_shape)

        function.cache_clear()
        self.assertEqual(self.cache_shape(), (0, 0, None, 0))

    def test_signature_documentation_annotations_and_exports(self):
        function = torch.get_device_module
        wrapped = function.__wrapped__

        self.assertEqual(
            str(inspect.signature(function)),
            "(device: torch_rs.device | str | None = None)",
        )
        expected_annotations = {"device": torch.device | str | None}
        self.assertEqual(inspect.get_annotations(function), expected_annotations)
        self.assertEqual(
            typing.get_type_hints(function),
            expected_annotations,
        )
        if sys.version_info >= (3, 14):
            self.assertFalse(hasattr(function, "__annotations__"))
            self.assertTrue(hasattr(function, "__annotate__"))
        else:
            self.assertEqual(function.__annotations__, expected_annotations)
            self.assertFalse(hasattr(function, "__annotate__"))
        self.assertEqual(function.__name__, "get_device_module")
        self.assertEqual(function.__qualname__, "get_device_module")
        self.assertEqual(function.__module__, "torch_rs")
        self.assertIs(inspect.getmodule(function), torch)
        self.assertEqual(
            inspect.cleandoc(function.__doc__), inspect.cleandoc(FUNCTION_DOC)
        )
        self.assertFalse(hasattr(function, "__defaults__"))
        self.assertFalse(hasattr(function, "__kwdefaults__"))
        self.assertFalse(hasattr(function, "__text_signature__"))

        self.assertIs(type(wrapped), types.FunctionType)
        self.assertEqual(inspect.signature(wrapped), inspect.signature(function))
        self.assertEqual(inspect.get_annotations(wrapped), expected_annotations)
        self.assertEqual(wrapped.__defaults__, (None,))
        self.assertIsNone(wrapped.__kwdefaults__)

        self.assertEqual(torch.__all__.count("get_device_module"), 1)
        direct_import = {}
        wildcard_import = {}
        exec("from torch_rs import get_device_module", direct_import)
        exec("from torch_rs import *", wildcard_import)
        self.assertIs(direct_import["get_device_module"], function)
        self.assertIs(wildcard_import["get_device_module"], function)
        self.assertFalse(hasattr(torch._C, "get_device_module"))
        self.assertNotIn("get_device_module", torch._C.__all__)

    def test_copy_and_pickle_preserve_the_global_cached_wrapper(self):
        function = torch.get_device_module

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs", payload)
                self.assertIn(b"get_device_module", payload)
                self.assertIs(pickle.loads(payload), function)

        for operation in (
            copy.copy,
            copy.deepcopy,
            lambda module: pickle.loads(pickle.dumps(module)),
        ):
            with self.subTest(operation=operation):
                with self.assertRaisesRegex(
                    TypeError, "^cannot pickle 'module' object$"
                ):
                    operation(torch.get_device_module("cpu"))

    def test_invalid_values_errors_and_failed_call_cache_accounting(self):
        class StringLike:
            def __str__(self):
                return "cpu"

        class BadString:
            def __str__(self):
                raise ValueError("bad device text")

        function = torch.get_device_module
        hashable_cases = (
            (
                lambda: function(None, None),
                TypeError,
                "get_device_module() takes from 0 to 1 positional arguments but 2 were given",
            ),
            (
                lambda: function(foo=None),
                TypeError,
                "get_device_module() got an unexpected keyword argument 'foo'",
            ),
            (
                lambda: function(None, device=None),
                TypeError,
                "get_device_module() got multiple values for argument 'device'",
            ),
            (
                lambda: function(0),
                RuntimeError,
                "Invalid value of device '0', expect torch.device, str, or None",
            ),
            (
                lambda: function(False),
                RuntimeError,
                "Invalid value of device 'False', expect torch.device, str, or None",
            ),
            (
                lambda: function(1.5),
                RuntimeError,
                "Invalid value of device '1.5', expect torch.device, str, or None",
            ),
            (
                lambda: function(("cpu",)),
                RuntimeError,
                "Invalid value of device '('cpu',)', expect torch.device, str, or None",
            ),
            (
                lambda: function(StringLike()),
                RuntimeError,
                "Invalid value of device 'cpu', expect torch.device, str, or None",
            ),
            (lambda: function(BadString()), ValueError, "bad device text"),
            (lambda: function(""), RuntimeError, "Device string must not be empty"),
            (
                lambda: function("cpu:"),
                RuntimeError,
                "Invalid device string: 'cpu:'",
            ),
            (
                lambda: function("cpu:-1"),
                RuntimeError,
                "Invalid device string: 'cpu:-1'",
            ),
            (
                lambda: function("cpu:01"),
                RuntimeError,
                "Invalid device string: 'cpu:01'",
            ),
            (
                lambda: function("cpu:2147483648"),
                RuntimeError,
                "Could not parse device index '2147483648' in device string 'cpu:2147483648'",
            ),
        )
        for call, error_type, message in hashable_cases:
            with self.subTest(message=message):
                before = self.cache_shape()
                with self.assertRaises(error_type) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                after = self.cache_shape()
                self.assertEqual(after[0], before[0])
                self.assertEqual(after[1], before[1] + 1)
                self.assertEqual(after[2], before[2])
                self.assertEqual(after[3], before[3])

        for value, message in (
            ([], "unhashable type: 'list'"),
            ({}, "unhashable type: 'dict'"),
            (set(), "unhashable type: 'set'"),
        ):
            with self.subTest(value_type=type(value).__name__):
                before = self.cache_shape()
                with self.assertRaises(TypeError) as raised:
                    function(value)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertEqual(self.cache_shape(), before)

    def test_default_stays_cpu_without_importing_accelerator_modules(self):
        function = torch.get_device_module
        with mock.patch.dict(
            os.environ,
            {"CUDA_VISIBLE_DEVICES": "0", "PYTORCH_NVML_BASED_CUDA_CHECK": "1"},
            clear=True,
        ):
            self.assertIs(function(), torch.cpu)

        self.assertFalse(hasattr(torch, "cuda"))
        self.assertNotIn("torch_rs.cuda", sys.modules)
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("torch_rs.cuda")

        script = r"""
import os
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
os.environ.update(
    CUDA_VISIBLE_DEVICES="0",
    PYTORCH_NVML_BASED_CUDA_CHECK="1",
)
import torch_rs

assert torch_rs.get_device_module() is torch_rs.cpu
assert torch_rs.get_device_module("cpu:7") is torch_rs.cpu
assert not hasattr(torch_rs, "cuda")
assert "torch_rs.cuda" not in sys.modules
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
"""
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


if __name__ == "__main__":
    unittest.main()
