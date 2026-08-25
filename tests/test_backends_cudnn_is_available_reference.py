import copy
import importlib
import inspect
import pickle
import pickletools
import re
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
class CudnnReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "backends.cudnn differentials require pinned "
                "PyTorch 2.13.0"
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

    def fresh_cudnn_module(self, root):
        module_name = f"{root.__name__}.backends.cudnn"
        sys.modules.pop(module_name, None)
        if hasattr(root.backends, "cudnn"):
            del root.backends.cudnn
        module = importlib.import_module(module_name)
        root.backends.cudnn = module
        return module

    def test_function_metadata_and_identity_match_pytorch_2_13(self):
        actual_module = importlib.import_module("torch_rs.backends.cudnn")
        expected_module = importlib.import_module("torch.backends.cudnn")

        self.assertIs(torch.backends.cudnn, actual_module)
        self.assertIs(reference_torch.backends.cudnn, expected_module)
        self.assertIs(sys.modules[actual_module.__name__], actual_module)
        self.assertIs(sys.modules[expected_module.__name__], expected_module)
        self.assertEqual(type(actual_module).__name__, type(expected_module).__name__)
        self.assertEqual(
            type(actual_module).__module__.replace("torch_rs", "torch"),
            type(expected_module).__module__,
        )
        self.assertIsNone(actual_module.__doc__)
        self.assertEqual(actual_module.__doc__, expected_module.__doc__)
        self.assertEqual(
            hasattr(actual_module, "__all__"),
            hasattr(expected_module, "__all__"),
        )
        self.assertEqual(
            {name for name in vars(actual_module) if not name.startswith("_")},
            {name for name in vars(expected_module) if not name.startswith("_")},
        )
        self.assertIs(type(actual_module.m), types.ModuleType)
        self.assertIs(type(expected_module.m), types.ModuleType)

        for name in ("is_available", "version"):
            with self.subTest(name=name):
                actual = getattr(actual_module, name)
                expected = getattr(expected_module, name)
                self.assertIs(actual, getattr(actual_module.m, name))
                self.assertIs(expected, getattr(expected_module.m, name))
                self.assertIs(type(actual), types.FunctionType)
                self.assertIs(type(expected), types.FunctionType)
                self.assertEqual(
                    str(inspect.signature(actual)), str(inspect.signature(expected))
                )
                self.assertEqual(
                    inspect.get_annotations(actual), inspect.get_annotations(expected)
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
                self.assertEqual(actual.__doc__, expected.__doc__)
                self.assertEqual(actual.__defaults__, expected.__defaults__)
                self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
                self.assertEqual(actual.__dict__, expected.__dict__)
                self.assertEqual(
                    hasattr(actual, "__text_signature__"),
                    hasattr(expected, "__text_signature__"),
                )
                self.assertEqual(actual.__code__.co_names, expected.__code__.co_names)
                self.assertEqual(
                    actual.__code__.co_freevars, expected.__code__.co_freevars
                )
                self.assertEqual(
                    actual.__code__.co_cellvars, expected.__code__.co_cellvars
                )

    def test_imports_wildcards_copying_and_pickling_match_pytorch_2_13(self):
        actual_backends = importlib.import_module("torch_rs.backends")
        expected_backends = importlib.import_module("torch.backends")
        actual_module = importlib.import_module("torch_rs.backends.cudnn")
        expected_module = importlib.import_module("torch.backends.cudnn")

        self.assertIs(torch.backends, actual_backends)
        self.assertIs(reference_torch.backends, expected_backends)
        self.assertIs(actual_backends.cudnn, actual_module)
        self.assertIs(expected_backends.cudnn, expected_module)

        for package_name, module in (
            ("torch_rs", actual_module),
            ("torch", expected_module),
        ):
            backend_import = {}
            exec(f"from {package_name}.backends import cudnn", backend_import)
            self.assertIs(backend_import["cudnn"], module)
            for name in ("is_available", "version"):
                function_import = {}
                exec(
                    f"from {package_name}.backends.cudnn import {name}",
                    function_import,
                )
                self.assertIs(function_import[name], getattr(module, name))

        actual_parent_wildcard = {}
        expected_parent_wildcard = {}
        exec("from torch_rs.backends import *", actual_parent_wildcard)
        exec("from torch.backends import *", expected_parent_wildcard)
        self.assertEqual(
            {
                name
                for name in actual_parent_wildcard
                if not name.startswith("__")
            },
            {
                name
                for name in expected_parent_wildcard
                if name in {"cuda", "cudnn", "mkl", "nnpack", "openmp"}
            },
        )

        actual_child_wildcard = {}
        expected_child_wildcard = {}
        exec("from torch_rs.backends.cudnn import *", actual_child_wildcard)
        exec("from torch.backends.cudnn import *", expected_child_wildcard)
        self.assertEqual(
            {name for name in actual_child_wildcard if not name.startswith("__")},
            {name for name in expected_child_wildcard if not name.startswith("__")},
        )

        for name in ("is_available", "version"):
            actual = getattr(actual_module, name)
            expected = getattr(expected_module, name)
            with self.subTest(name=name):
                self.assertIs(copy.copy(actual), actual)
                self.assertIs(copy.copy(expected), expected)
                self.assertIs(copy.deepcopy(actual), actual)
                self.assertIs(copy.deepcopy(expected), expected)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                    self.assertIs(
                        pickle.loads(pickle.dumps(expected, protocol)), expected
                    )
                    self.assertEqual(
                        self.pickle_shape(actual, protocol),
                        self.pickle_shape(expected, protocol),
                    )
        for copier in (copy.copy, copy.deepcopy):
            errors = []
            for module in (actual_module, expected_module):
                try:
                    copier(module)
                except Exception as error:
                    errors.append((type(error), str(error)))
                else:
                    self.fail("a cuDNN module proxy unexpectedly supported copying")
            self.assertEqual(errors[0], errors[1])

    def reload_contract(self, root, function_name):
        parent = root.backends
        module = parent.cudnn
        old_function = getattr(module, function_name)
        namespace = module.__dict__
        reloaded = importlib.reload(module)
        new_function = getattr(module, function_name)

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
            self.fail("a stale cuDNN availability function remained pickleable")

        contract = (
            reloaded is module,
            module.__dict__ is namespace,
            parent.cudnn is module,
            sys.modules[module.__name__] is module,
            sys.modules[module.__name__] is reloaded,
            reloaded.m is module,
            old_function is not new_function,
            getattr(reloaded, function_name) is new_function,
            copy.copy(new_function) is new_function,
            copy.deepcopy(new_function) is new_function,
            pickle.loads(pickle.dumps(new_function)) is new_function,
            stale_pickle_error,
        )
        self.fresh_cudnn_module(root)
        return contract

    def test_reload_behavior_matches_pytorch_2_13(self):
        for name in ("is_available", "version"):
            with self.subTest(name=name):
                self.assertEqual(
                    self.reload_contract(torch, name),
                    self.reload_contract(reference_torch, name),
                )
                actual = getattr(torch.backends.cudnn, name)
                expected = getattr(reference_torch.backends.cudnn, name)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    self.assertEqual(
                        self.pickle_shape(actual, protocol),
                        self.pickle_shape(expected, protocol),
                    )

    def test_argument_errors_match_pytorch_2_13(self):
        for name in ("is_available", "version"):
            actual = getattr(torch.backends.cudnn, name)
            expected = getattr(reference_torch.backends.cudnn, name)
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
                with self.subTest(name=name, case=case):
                    self.assert_error_matches(actual_call, expected_call)

    def test_cudnn_enabled_h100_exposes_build_and_execution_boundary(self):
        if not reference_torch.backends.cudnn.is_available():
            self.skipTest("requires a cuDNN-built reference PyTorch")
        if not reference_torch.cuda.is_available():
            self.skipTest("requires a CUDA-visible reference PyTorch runtime")

        device_name = reference_torch.cuda.get_device_name(0)
        if "H100" not in device_name:
            self.skipTest(f"requires an NVIDIA H100, found {device_name}")

        self.assertIs(reference_torch._C._has_cudnn, True)
        self.assertIs(reference_torch.backends.cudnn.is_available(), True)
        self.assertIs(torch._C._has_cudnn, False)
        self.assertIs(torch.backends.cudnn.is_available(), False)
        self.assertGreater(reference_torch.backends.cudnn.version(), 0)
        self.assertIs(torch.backends.cudnn.version(), None)

        device = reference_torch.device("cuda", 0)
        source = reference_torch.arange(
            1.0,
            17.0,
            device=device,
        ).reshape(1, 1, 4, 4)
        kernel = reference_torch.ones((1, 1, 3, 3), device=device)
        result = reference_torch.nn.functional.conv2d(source, kernel)
        reference_torch.cuda.synchronize(device)
        self.assertEqual(result.cpu().tolist(), [[[[54.0, 63.0], [90.0, 99.0]]]])

        self.assertFalse(hasattr(torch.backends.cudnn, "flags"))
        self.assertFalse(hasattr(torch, "cuda"))
        self.assertFalse(hasattr(torch.Tensor, "cuda"))
        self.assertFalse(hasattr(torch.Tensor, "to"))
        with self.assertRaises(RuntimeError):
            torch.tensor([1.0], device="cuda:0")

    def test_configuration_and_execution_surface_remains_unsupported(self):
        actual = torch.backends.cudnn
        expected = reference_torch.backends.cudnn
        self.assertEqual(
            {name for name in vars(actual) if not name.startswith("_")},
            {"m"},
        )
        self.assertEqual(
            {name for name in vars(actual) if not name.startswith("_")},
            {name for name in vars(expected) if not name.startswith("_")},
        )
        for name in (
            "CUDNN_TENSOR_DTYPES",
            "allow_tf32",
            "benchmark",
            "benchmark_limit",
            "conv",
            "depthwise_kernel",
            "deterministic",
            "enabled",
            "flags",
            "fp32_precision",
            "is_acceptable",
            "rnn",
            "set_flags",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(actual, name))
                self.assertTrue(hasattr(expected, name))

        self.assertIs(actual.version(), None)
        expected_version = expected.version()
        if reference_torch._C._has_cudnn:
            self.assertIsInstance(expected_version, int)
        else:
            self.assertIs(expected_version, None)

        self.assertFalse(hasattr(torch, "cuda"))
        self.assertTrue(hasattr(reference_torch, "cuda"))


if __name__ == "__main__":
    unittest.main()
