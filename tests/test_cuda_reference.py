import copy
import importlib
import inspect
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


SUPPORTED = {"device_count", "is_available", "is_initialized"}


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

    def test_signature_annotations_and_identity_match_pytorch_2_13(self):
        actual_module = importlib.import_module("torch_rs.cuda")
        expected_module = importlib.import_module("torch.cuda")

        self.assertIs(torch.cuda, actual_module)
        self.assertIs(reference_torch.cuda, expected_module)
        self.assertIs(sys.modules["torch_rs.cuda"], actual_module)
        self.assertIs(sys.modules["torch.cuda"], expected_module)
        self.assertEqual(
            actual_module.__all__,
            [name for name in expected_module.__all__ if name in SUPPORTED],
        )

        for name in actual_module.__all__:
            with self.subTest(name=name):
                actual = getattr(actual_module, name)
                expected = getattr(expected_module, name)
                self.assertIs(type(actual), types.FunctionType)
                self.assertIs(type(expected), types.FunctionType)
                self.assertEqual(
                    str(inspect.signature(actual)), str(inspect.signature(expected))
                )
                self.assertEqual(
                    inspect.get_annotations(actual),
                    inspect.get_annotations(expected),
                )
                self.assertEqual(
                    typing.get_type_hints(actual), typing.get_type_hints(expected)
                )
                self.assertEqual(actual.__name__, expected.__name__)
                self.assertEqual(actual.__qualname__, expected.__qualname__)
                self.assertEqual(
                    actual.__module__.replace("torch_rs", "torch"),
                    expected.__module__,
                )
                self.assertIs(inspect.getmodule(actual), actual_module)
                self.assertIs(inspect.getmodule(expected), expected_module)
                if name == "is_initialized":
                    self.assertEqual(actual.__doc__, expected.__doc__)
                self.assertEqual(actual.__defaults__, expected.__defaults__)
                self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
                self.assertEqual(actual.__dict__, expected.__dict__)
                self.assertEqual(
                    hasattr(actual, "__text_signature__"),
                    hasattr(expected, "__text_signature__"),
                )

    def test_imports_exports_copy_and_pickle_match_supported_scope(self):
        actual_module = torch.cuda
        expected_module = reference_torch.cuda

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
            for name in sorted(SUPPORTED):
                with self.subTest(package=package_name, name=name):
                    self.assertIs(direct_import[name], getattr(module, name))
                    self.assertIs(module_wildcard[name], getattr(module, name))
                    self.assertNotIn(name, top_level_wildcard)
            self.assertNotIn("cuda", top_level_wildcard)

        actual_wildcard = {}
        expected_wildcard = {}
        exec("from torch_rs.cuda import *", actual_wildcard)
        exec("from torch.cuda import *", expected_wildcard)
        self.assertEqual(
            {name for name in actual_wildcard if not name.startswith("__")},
            {
                name
                for name in expected_wildcard
                if name in SUPPORTED
            },
        )

        for name in sorted(SUPPORTED):
            with self.subTest(name=name):
                actual = getattr(actual_module, name)
                expected = getattr(expected_module, name)
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

    def test_argument_errors_match_pytorch_2_13(self):
        cases = (
            ("is_available", (None,), {}),
            ("is_available", (None, None), {}),
            ("is_available", (), {"enabled": True}),
            ("is_available", (None,), {"enabled": True}),
            ("device_count", (None,), {}),
            ("device_count", (None, None), {}),
            ("device_count", (), {"device": True}),
            ("device_count", (None,), {"device": True}),
            ("is_initialized", (None,), {}),
            ("is_initialized", (None, None), {}),
            ("is_initialized", (), {"enabled": True}),
            ("is_initialized", (None,), {"enabled": True}),
        )

        for name, args, kwargs in cases:
            with self.subTest(name=name, args=args, kwargs=kwargs):
                actual = getattr(torch.cuda, name)
                expected = getattr(reference_torch.cuda, name)
                self.assert_error_matches(
                    lambda: actual(*args, **kwargs),
                    lambda: expected(*args, **kwargs),
                )

    def test_return_values_keep_the_cpu_build_contract(self):
        self.assertIs(torch.cuda.is_available(), False)
        self.assertIs(type(torch.cuda.device_count()), int)
        self.assertEqual(torch.cuda.device_count(), 0)
        self.assertIs(torch.cuda.is_initialized(), False)
        self.assertFalse(hasattr(torch.cuda, "_initialized"))
        self.assertFalse(hasattr(torch.cuda, "_cached_device_count"))

    def test_fresh_subprocess_initialization_and_reload_behavior(self):
        script = r'''
import copy
import importlib
import pickle
import re
import sys

import torch
import torch_rs

if torch.__version__.split("+", 1)[0] != "2.13.0":
    raise AssertionError(f"expected PyTorch 2.13.0, got {torch.__version__}")

assert torch.cuda.is_initialized() is False
reference_available = torch.cuda.is_available()
reference_count = torch.cuda.device_count()
assert type(reference_available) is bool
assert type(reference_count) is int
assert torch.cuda.is_initialized() is False

assert torch_rs.cuda.is_available() is False
assert type(torch_rs.cuda.device_count()) is int
assert torch_rs.cuda.device_count() == 0
assert torch_rs.cuda.is_initialized() is False
assert not hasattr(torch_rs.cuda, "_initialized")
assert not hasattr(torch_rs.cuda, "_cached_device_count")

SUPPORTED = ("device_count", "is_available", "is_initialized")

def normalize(message):
    return re.sub(r"0x[0-9a-fA-F]+", "0x...", message).replace("torch_rs", "torch")

def reload_contract(root):
    module = root.cuda
    old_functions = {name: getattr(module, name) for name in SUPPORTED}
    namespace = module.__dict__
    reloaded = importlib.reload(module)
    new_functions = {name: getattr(module, name) for name in SUPPORTED}
    stale_pickle_errors = []
    for name in sorted(SUPPORTED):
        try:
            pickle.dumps(old_functions[name])
        except Exception as error:
            stale_pickle_errors.append((type(error).__name__, normalize(str(error))))
        else:
            raise AssertionError(f"stale {root.__name__}.cuda.{name} stayed pickleable")
    return (
        reloaded is module,
        module.__dict__ is namespace,
        root.cuda is module,
        sys.modules[module.__name__] is module,
        tuple(old_functions[name] is not new_functions[name] for name in SUPPORTED),
        tuple(
            copy.copy(new_functions[name]) is new_functions[name]
            for name in SUPPORTED
        ),
        tuple(
            copy.deepcopy(new_functions[name]) is new_functions[name]
            for name in SUPPORTED
        ),
        tuple(
            pickle.loads(pickle.dumps(new_functions[name])) is new_functions[name]
            for name in SUPPORTED
        ),
        tuple(stale_pickle_errors),
    )

assert reload_contract(torch_rs) == reload_contract(torch)

assert torch_rs.cuda.is_available() is False
assert type(torch_rs.cuda.device_count()) is int
assert torch_rs.cuda.device_count() == 0
assert torch_rs.cuda.is_initialized() is False
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


if __name__ == "__main__":
    unittest.main()
