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


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CudaReferenceTests(unittest.TestCase):
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

    def test_is_initialized_signature_and_errors_match_pytorch_2_13(self):
        actual_module = importlib.import_module("torch_rs.cuda")
        expected_module = importlib.import_module("torch.cuda")
        actual = actual_module.is_initialized
        expected = expected_module.is_initialized

        self.assertIs(torch.cuda, actual_module)
        self.assertIs(reference_torch.cuda, expected_module)
        self.assertIs(sys.modules["torch_rs.cuda"], actual_module)
        self.assertIs(sys.modules["torch.cuda"], expected_module)
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

    def test_imports_wildcards_copy_pickle_and_reload_match_supported_scope(self):
        actual_cuda = torch.cuda
        expected_cuda = reference_torch.cuda
        supported = {
            "device_count",
            "is_available",
            "is_initialized",
            "memory_reserved",
        }

        self.assertEqual(
            actual_cuda.__all__,
            [name for name in expected_cuda.__all__ if name in supported],
        )
        for name in ("cuda", *sorted(supported)):
            with self.subTest(top_level_export=name):
                self.assertEqual(
                    torch.__all__.count(name), reference_torch.__all__.count(name)
                )

        actual_package_import = {}
        expected_package_import = {}
        exec("from torch_rs import cuda", actual_package_import)
        exec("from torch import cuda", expected_package_import)
        self.assertIs(actual_package_import["cuda"], actual_cuda)
        self.assertIs(expected_package_import["cuda"], expected_cuda)

        actual_direct_import = {}
        expected_direct_import = {}
        exec(
            "from torch_rs.cuda import device_count, is_available, is_initialized, memory_reserved",
            actual_direct_import,
        )
        exec(
            "from torch.cuda import device_count, is_available, is_initialized, memory_reserved",
            expected_direct_import,
        )
        for name in supported:
            with self.subTest(direct_import=name):
                self.assertIs(actual_direct_import[name], getattr(actual_cuda, name))
                self.assertIs(expected_direct_import[name], getattr(expected_cuda, name))

        actual_wildcard = {}
        expected_wildcard = {}
        exec("from torch_rs.cuda import *", actual_wildcard)
        exec("from torch.cuda import *", expected_wildcard)
        self.assertEqual(
            {name for name in actual_wildcard if not name.startswith("__")},
            supported,
        )
        for name in supported:
            with self.subTest(wildcard=name):
                self.assertIs(actual_wildcard[name], getattr(actual_cuda, name))
                self.assertIs(expected_wildcard[name], getattr(expected_cuda, name))

        for module, functions in (
            (torch, (actual_cuda.is_initialized, actual_cuda.memory_reserved)),
            (
                reference_torch,
                (expected_cuda.is_initialized, expected_cuda.memory_reserved),
            ),
        ):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("cuda", namespace)
            self.assertNotIn("is_initialized", namespace)
            self.assertNotIn("memory_reserved", namespace)
            for function in functions:
                with self.subTest(module=module.__name__, function=function.__name__):
                    self.assertIs(copy.copy(function), function)
                    self.assertIs(copy.deepcopy(function), function)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                for actual, expected in (
                    (actual_cuda.is_initialized, expected_cuda.is_initialized),
                    (actual_cuda.memory_reserved, expected_cuda.memory_reserved),
                ):
                    with self.subTest(function=actual.__name__):
                        self.assertIs(
                            pickle.loads(pickle.dumps(actual, protocol)),
                            actual,
                        )
                        self.assertIs(
                            pickle.loads(pickle.dumps(expected, protocol)),
                            expected,
                        )
                self.assertEqual(
                    self.pickle_shape(actual_cuda.is_initialized, protocol),
                    self.pickle_shape(expected_cuda.is_initialized, protocol),
                )

        actual_old = actual_cuda.is_initialized
        expected_old = expected_cuda.is_initialized
        self.assertIs(importlib.reload(actual_cuda), actual_cuda)
        self.assertIs(importlib.reload(expected_cuda), expected_cuda)
        self.assertIsNot(actual_cuda.is_initialized, actual_old)
        self.assertIsNot(expected_cuda.is_initialized, expected_old)
        self.assertIs(torch.cuda, actual_cuda)
        self.assertIs(reference_torch.cuda, expected_cuda)
        self.assertIs(sys.modules["torch_rs.cuda"], actual_cuda)
        self.assertIs(sys.modules["torch.cuda"], expected_cuda)
        self.assertIs(actual_cuda.is_initialized(), False)
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(actual_old)
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(expected_old)

    def test_memory_reserved_signature_and_call_errors_match_pytorch_2_13(self):
        actual_module = importlib.import_module("torch_rs.cuda")
        expected_module = importlib.import_module("torch.cuda")
        actual = actual_module.memory_reserved
        expected = expected_module.memory_reserved

        self.assertIs(torch.cuda, actual_module)
        self.assertIs(reference_torch.cuda, expected_module)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual)), str(inspect.signature(expected))
        )
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        for function in (actual, expected):
            with self.subTest(function=function.__module__):
                with self.assertRaises(NameError):
                    typing.get_type_hints(function)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(actual.__module__, "torch_rs.cuda")
        self.assertEqual(expected.__module__, "torch.cuda.memory")
        self.assertIs(inspect.getmodule(actual), actual_module)
        self.assertEqual(
            inspect.cleandoc(actual.__doc__),
            inspect.cleandoc(expected.__doc__),
        )
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )

        cases = (
            (lambda: actual(None, None), lambda: expected(None, None)),
            (lambda: actual(enabled=True), lambda: expected(enabled=True)),
            (
                lambda: actual(None, enabled=True),
                lambda: expected(None, enabled=True),
            ),
            (
                lambda: actual(None, device=None),
                lambda: expected(None, device=None),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_memory_reserved_cpu_build_zero_probe_matches_uninitialized_pytorch(self):
        actual = torch.cuda.memory_reserved

        class ExplodingDevice:
            def __getattribute__(self, name):
                raise AssertionError(f"device attribute was read: {name}")

            def __repr__(self):
                raise AssertionError("device repr was read")

            def __str__(self):
                raise AssertionError("device str was read")

        device_forms = (
            ("omitted", (), {}),
            ("none positional", (None,), {}),
            ("none keyword", (), {"device": None}),
            ("cuda string", ("cuda",), {}),
            ("cuda index string", ("cuda:0",), {}),
            ("cuda negative string", ("cuda:-1",), {}),
            ("cpu string", ("cpu",), {}),
            ("cpu index string", ("cpu:0",), {}),
            ("empty string", ("",), {}),
            ("unknown string", ("banana",), {}),
            ("int zero", (0,), {}),
            ("negative int", (-1,), {}),
            ("large int", (sys.maxsize,), {}),
            ("cpu device", (torch.device("cpu"),), {}),
            ("indexed cpu device", (torch.device("cpu:0"),), {}),
            ("object", (object(),), {}),
            ("unreadable object", (ExplodingDevice(),), {}),
        )
        for label, args, kwargs in device_forms:
            with self.subTest(label=label):
                result = actual(*args, **kwargs)
                self.assertIs(type(result), int)
                self.assertEqual(result, 0)
                self.assertIs(torch.cuda.is_available(), False)
                self.assertEqual(torch.cuda.device_count(), 0)
                self.assertIs(torch.cuda.is_initialized(), False)

        script = r"""
import json
import sys
import torch

assert torch.__version__.split("+")[0] == "2.13.0"
assert torch.cuda.is_initialized() is False

class ExplodingDevice:
    def __getattribute__(self, name):
        raise AssertionError(f"device attribute was read: {name}")

    def __repr__(self):
        raise AssertionError("device repr was read")

    def __str__(self):
        raise AssertionError("device str was read")

device_forms = (
    ("omitted", (), {}),
    ("none positional", (None,), {}),
    ("none keyword", (), {"device": None}),
    ("cuda string", ("cuda",), {}),
    ("cuda index string", ("cuda:0",), {}),
    ("cuda negative string", ("cuda:-1",), {}),
    ("cpu string", ("cpu",), {}),
    ("cpu index string", ("cpu:0",), {}),
    ("empty string", ("",), {}),
    ("unknown string", ("banana",), {}),
    ("int zero", (0,), {}),
    ("negative int", (-1,), {}),
    ("large int", (sys.maxsize,), {}),
    ("cpu device", (torch.device("cpu"),), {}),
    ("indexed cpu device", (torch.device("cpu:0"),), {}),
    ("object", (object(),), {}),
    ("unreadable object", (ExplodingDevice(),), {}),
)
observed = []
for label, args, kwargs in device_forms:
    result = torch.cuda.memory_reserved(*args, **kwargs)
    observed.append([label, type(result) is int, result])

print(json.dumps(
    {
        "initialized": torch.cuda.is_initialized(),
        "observed": observed,
    },
    sort_keys=True,
))
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
        expected = [[label, True, 0] for label, _, _ in device_forms]
        self.assertEqual(
            json.loads(completed.stdout),
            {"initialized": False, "observed": expected},
        )

    def test_cpu_build_probe_values_are_static_without_changing_cuda_runtime_state(self):
        cuda = torch.cuda
        self.assertIs(cuda.is_initialized(), False)
        self.assertIs(cuda.is_available(), False)
        self.assertEqual(cuda.device_count(), 0)
        self.assertEqual(cuda.memory_reserved(), 0)
        self.assertEqual(cuda.memory_reserved("cuda:0"), 0)
        self.assertIs(cuda.is_initialized(), False)
        self.assertFalse(hasattr(cuda, "_initialized"))
        self.assertFalse(hasattr(cuda, "_cached_device_count"))

        script = r"""
import torch

assert torch.cuda.is_initialized() is False
assert type(torch.cuda.is_available()) is bool
assert type(torch.cuda.device_count()) is int
assert type(torch.cuda.memory_reserved()) is int
assert torch.cuda.memory_reserved() == 0
assert torch.cuda.memory_reserved("cuda:0") == 0
assert torch.cuda.is_initialized() is False
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
