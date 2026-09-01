import copy
import importlib
import inspect
import pickle
import sys
import types
import typing
import unittest
import warnings

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class MkldnnAvailabilityReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "backends.mkldnn.is_available differentials require pinned "
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

    def normalize(self, value):
        return str(value).replace("torch_rs", "torch")

    def exception_contract(self, callback):
        try:
            callback()
        except Exception as error:
            return type(error).__name__, self.normalize(error)
        self.fail("expected an exception")

    def copy_contract(self, function):
        copied = copy.copy(function)
        return (
            copied == function,
            copied is function,
            copied.__self__ is function.__self__,
            type(copied).__name__,
        )

    def deprecated_has_mkldnn_contract(self, root):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = root.has_mkldnn
        return (
            type(value).__name__,
            name := "has_mkldnn",
            name in vars(root),
            root.__all__.count(name),
            len(caught),
            type(caught[0].message).__name__,
            self.normalize(caught[0].message),
        )

    def test_value_and_native_flag_match_pytorch_2_13_shape(self):
        actual_module = importlib.import_module("torch_rs.backends.mkldnn")
        expected_module = importlib.import_module("torch.backends.mkldnn")
        actual = actual_module.is_available()
        expected = expected_module.is_available()

        self.assertIs(type(actual), bool)
        self.assertIs(type(expected), bool)
        self.assertIs(actual, torch._C._has_mkldnn)
        self.assertIs(expected, reference_torch._C._has_mkldnn)
        self.assertIs(actual, False)
        self.assertIs(torch.has_mkl, False)
        self.assertIs(torch.has_lapack, False)
        self.assertEqual(
            self.deprecated_has_mkldnn_contract(torch),
            (
                "bool",
                "has_mkldnn",
                False,
                0,
                1,
                "UserWarning",
                "'has_mkldnn' is deprecated, please use "
                "'torch.backends.mkldnn.is_available()'",
            ),
        )
        self.assertEqual(
            self.deprecated_has_mkldnn_contract(torch),
            self.deprecated_has_mkldnn_contract(reference_torch),
        )

    def test_signature_documentation_and_identity_match_pytorch_2_13(self):
        actual_module = importlib.import_module("torch_rs.backends.mkldnn")
        expected_module = importlib.import_module("torch.backends.mkldnn")
        actual = actual_module.is_available
        expected = expected_module.is_available

        self.assertIs(torch.backends.mkldnn, actual_module)
        self.assertIs(reference_torch.backends.mkldnn, expected_module)
        self.assertIs(sys.modules[actual_module.__name__], actual_module)
        self.assertIs(sys.modules[expected_module.__name__], expected_module)
        self.assertIsInstance(actual_module, types.ModuleType)
        self.assertIsInstance(expected_module, types.ModuleType)
        self.assertEqual(type(actual_module).__name__, type(expected_module).__name__)
        self.assertEqual(actual_module.__doc__, expected_module.__doc__)
        self.assertEqual(
            hasattr(actual_module, "__all__"),
            hasattr(expected_module, "__all__"),
        )
        self.assertEqual(
            {name for name in vars(actual_module) if not name.startswith("_")},
            {
                name
                for name in vars(expected_module)
                if not name.startswith("_") and name in {"m"}
            },
        )
        self.assertIs(actual_module.m.torch, torch)
        self.assertIs(expected_module.m.torch, reference_torch)

        self.assertIs(type(actual), types.MethodType)
        self.assertIs(type(expected), types.MethodType)
        self.assertIs(actual.__self__, actual_module)
        self.assertIs(expected.__self__, expected_module)
        self.assertEqual(str(inspect.signature(actual)), str(inspect.signature(expected)))
        self.assertEqual(inspect.get_annotations(actual), inspect.get_annotations(expected))
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
        self.assertEqual(actual.__func__.__code__.co_names, expected.__func__.__code__.co_names)
        self.assertEqual(
            actual.__func__.__code__.co_freevars,
            expected.__func__.__code__.co_freevars,
        )
        self.assertEqual(
            actual.__func__.__code__.co_cellvars,
            expected.__func__.__code__.co_cellvars,
        )

    def test_imports_wildcards_copying_and_pickling_match_supported_scope(self):
        actual_backends = importlib.import_module("torch_rs.backends")
        expected_backends = importlib.import_module("torch.backends")
        actual_module = importlib.import_module("torch_rs.backends.mkldnn")
        expected_module = importlib.import_module("torch.backends.mkldnn")
        actual = actual_module.is_available
        expected = expected_module.is_available
        supported_backends = {
            "cpu",
            "cuda",
            "cusparselt",
            "cudnn",
            "kleidiai",
            "m",
            "mha",
            "mkl",
            "mkldnn",
            "nnpack",
            "openmp",
        }

        for package_name, module, function in (
            ("torch_rs", actual_module, actual),
            ("torch", expected_module, expected),
        ):
            backend_import = {}
            function_import = {}
            exec(f"from {package_name}.backends import mkldnn", backend_import)
            exec(
                f"from {package_name}.backends.mkldnn import is_available",
                function_import,
            )
            self.assertIs(backend_import["mkldnn"], module)
            self.assertEqual(function_import["is_available"], function)
            self.assertIs(function_import["is_available"].__self__, module)

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
                if name in supported_backends
            },
        )

        actual_child_wildcard = {}
        expected_child_wildcard = {}
        exec("from torch_rs.backends.mkldnn import *", actual_child_wildcard)
        exec("from torch.backends.mkldnn import *", expected_child_wildcard)
        self.assertEqual(
            {name for name in actual_child_wildcard if not name.startswith("__")},
            {
                name
                for name in expected_child_wildcard
                if name in {"m"}
            },
        )
        self.assertIs(actual_child_wildcard["m"], actual_module.m)

        for root in (torch, reference_torch):
            namespace = {}
            exec(f"from {root.__name__} import *", namespace)
            self.assertNotIn("backends", namespace)
            self.assertNotIn("mkldnn", namespace)
            self.assertNotIn("has_mkldnn", namespace)
            self.assertFalse(hasattr(root, "mkldnn"))

        self.assertEqual(self.copy_contract(actual), self.copy_contract(expected))
        self.assertEqual(
            self.exception_contract(lambda: copy.deepcopy(actual)),
            self.exception_contract(lambda: copy.deepcopy(expected)),
        )
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertEqual(
                    self.exception_contract(lambda: pickle.dumps(actual, protocol)),
                    self.exception_contract(lambda: pickle.dumps(expected, protocol)),
                )

    def reload_contract(self, root):
        parent = root.backends
        module = importlib.import_module(f"{root.__name__}.backends.mkldnn")
        self.assertIs(parent.mkldnn, module)
        old_function = module.is_available
        reloaded = importlib.reload(module)
        current_module = importlib.import_module(f"{root.__name__}.backends.mkldnn")
        setattr(parent, "mkldnn", current_module)
        new_function = current_module.is_available
        native_flag = root._C._has_mkldnn

        return (
            reloaded is current_module,
            sys.modules[current_module.__name__] is current_module,
            old_function is not new_function,
            new_function() is native_flag,
            type(new_function).__name__,
            type(current_module).__name__,
            self.exception_contract(lambda: pickle.dumps(old_function)),
            self.exception_contract(lambda: pickle.dumps(new_function)),
        )

    def test_reload_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_contract(torch),
            self.reload_contract(reference_torch),
        )
        actual = torch.backends.mkldnn.is_available
        expected = reference_torch.backends.mkldnn.is_available
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertEqual(
                    self.exception_contract(lambda: pickle.dumps(actual, protocol)),
                    self.exception_contract(lambda: pickle.dumps(expected, protocol)),
                )

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.backends.mkldnn.is_available
        expected = reference_torch.backends.mkldnn.is_available
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

    def test_unsupported_mkldnn_surface_is_bounded_to_the_probe(self):
        actual = torch.backends.mkldnn
        expected = reference_torch.backends.mkldnn

        self.assertEqual(
            {name for name in vars(actual) if not name.startswith("_")},
            {
                name
                for name in vars(expected)
                if not name.startswith("_") and name in {"m"}
            },
        )
        self.assertIn("is_available", vars(expected.m))
        for name in (
            "VERBOSE_OFF",
            "VERBOSE_ON",
            "VERBOSE_ON_CREATION",
            "enabled",
            "flags",
            "set_flags",
            "verbose",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(actual, name))
        self.assertFalse(hasattr(torch.Tensor, "to_mkldnn"))
        self.assertTrue(hasattr(reference_torch.Tensor, "to_mkldnn"))
        self.assertIs(torch.tensor([1.0]).is_mkldnn, False)


if __name__ == "__main__":
    unittest.main()
