import copy
import importlib
import inspect
import pickle
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
class MkldnnIsAvailableReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "backends.mkldnn.is_available differentials require pinned "
                "PyTorch 2.13.0"
            )

    def normalize(self, value):
        value = str(value).replace("torch_rs", "torch")
        return re.sub(r"0x[0-9a-fA-F]+", "0x...", value)

    def error_contract(self, call):
        try:
            call()
        except Exception as error:
            return (
                type(error).__name__,
                self.normalize(str(error)),
                tuple(
                    self.normalize(argument)
                    if isinstance(argument, str)
                    else argument
                    for argument in error.args
                ),
            )
        self.fail("operation unexpectedly succeeded")

    def assert_error_matches(self, actual_call, expected_call):
        actual = self.error_contract(actual_call)
        expected = self.error_contract(expected_call)
        self.assertEqual(actual, expected)

    def fresh_mkldnn_module(self, root):
        module_name = f"{root.__name__}.backends.mkldnn"
        sys.modules.pop(module_name, None)
        if hasattr(root.backends, "mkldnn"):
            del root.backends.mkldnn
        module = importlib.import_module(module_name)
        root.backends.mkldnn = module
        return module

    def method_metadata(self, method):
        return (
            type(method).__name__,
            str(inspect.signature(method)),
            inspect.get_annotations(method),
            typing.get_type_hints(method),
            method.__name__,
            method.__qualname__,
            self.normalize(method.__module__),
            method.__doc__,
            method.__defaults__,
            method.__kwdefaults__,
            method.__dict__,
            hasattr(method, "__text_signature__"),
            method.__code__.co_names,
            method.__code__.co_freevars,
            method.__code__.co_cellvars,
        )

    def test_values_are_exact_build_specific_native_flags(self):
        actual_module = torch.backends.mkldnn
        expected_module = reference_torch.backends.mkldnn
        actual = actual_module.is_available()
        expected = expected_module.is_available()

        self.assertIs(type(actual), bool)
        self.assertIs(type(expected), bool)
        self.assertIs(actual, False)
        self.assertIs(actual, torch._C._has_mkldnn)
        self.assertIs(expected, reference_torch._C._has_mkldnn)
        self.assertFalse(hasattr(torch, "_has_mkldnn"))
        self.assertFalse(hasattr(reference_torch, "_has_mkldnn"))

    def test_signature_documentation_and_identity_match_pytorch_2_13(self):
        actual_module = importlib.import_module("torch_rs.backends.mkldnn")
        expected_module = importlib.import_module("torch.backends.mkldnn")
        actual = actual_module.is_available
        expected = expected_module.is_available

        self.assertIs(torch.backends.mkldnn, actual_module)
        self.assertIs(reference_torch.backends.mkldnn, expected_module)
        self.assertIs(sys.modules[actual_module.__name__], actual_module)
        self.assertIs(sys.modules[expected_module.__name__], expected_module)
        self.assertEqual(type(actual_module).__name__, type(expected_module).__name__)
        self.assertEqual(
            self.normalize(type(actual_module).__module__),
            type(expected_module).__module__,
        )
        self.assertIsInstance(actual_module, types.ModuleType)
        self.assertIsInstance(expected_module, types.ModuleType)
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
        self.assertIsNot(actual_module.m, actual_module)
        self.assertIsNot(expected_module.m, expected_module)

        self.assertEqual(self.method_metadata(actual), self.method_metadata(expected))
        self.assertIs(actual.__self__, actual_module)
        self.assertIs(expected.__self__, expected_module)
        self.assertIs(actual.__func__, type(actual_module).is_available)
        self.assertIs(expected.__func__, type(expected_module).is_available)
        self.assertIsNot(actual_module.is_available, actual_module.is_available)
        self.assertIsNot(expected_module.is_available, expected_module.is_available)
        self.assertEqual(actual_module.is_available, actual_module.is_available)
        self.assertEqual(expected_module.is_available, expected_module.is_available)

        actual_function = actual_module.m.is_available
        expected_function = expected_module.m.is_available
        self.assertEqual(
            self.method_metadata(actual_function),
            self.method_metadata(expected_function),
        )
        self.assertEqual(actual_function.__doc__, expected_function.__doc__)
        self.assertEqual(
            actual_function.__code__.co_names,
            expected_function.__code__.co_names,
        )

    def test_imports_wildcards_copying_and_pickling_match_pytorch_2_13(self):
        actual_backends = importlib.import_module("torch_rs.backends")
        expected_backends = importlib.import_module("torch.backends")
        actual_module = importlib.import_module("torch_rs.backends.mkldnn")
        expected_module = importlib.import_module("torch.backends.mkldnn")
        actual = actual_module.is_available
        expected = expected_module.is_available

        for package_name, module, method in (
            ("torch_rs", actual_module, actual),
            ("torch", expected_module, expected),
        ):
            backend_import = {}
            method_import = {}
            exec(f"from {package_name}.backends import mkldnn", backend_import)
            exec(
                f"from {package_name}.backends.mkldnn import is_available",
                method_import,
            )
            self.assertIs(backend_import["mkldnn"], module)
            self.assertEqual(method_import["is_available"], method)
            self.assertIs(method_import["is_available"].__self__, module)
            self.assertIs(method_import["is_available"].__func__, method.__func__)

        actual_parent_wildcard = {}
        expected_parent_wildcard = {}
        exec("from torch_rs.backends import *", actual_parent_wildcard)
        exec("from torch.backends import *", expected_parent_wildcard)
        self.assertEqual(
            {name for name in actual_parent_wildcard if not name.startswith("__")},
            {
                name
                for name in expected_parent_wildcard
                if name in {"cuda", "cudnn", "mkl", "mkldnn", "nnpack", "openmp"}
            },
        )

        actual_child_wildcard = {}
        expected_child_wildcard = {}
        exec("from torch_rs.backends.mkldnn import *", actual_child_wildcard)
        exec("from torch.backends.mkldnn import *", expected_child_wildcard)
        self.assertEqual(
            {name for name in actual_child_wildcard if not name.startswith("__")},
            {name for name in expected_child_wildcard if not name.startswith("__")},
        )

        actual_copy = copy.copy(actual)
        expected_copy = copy.copy(expected)
        self.assertIsNot(actual_copy, actual)
        self.assertIsNot(expected_copy, expected)
        self.assertEqual(actual_copy, actual)
        self.assertEqual(expected_copy, expected)
        self.assertIs(actual_copy.__self__, actual_module)
        self.assertIs(expected_copy.__self__, expected_module)
        self.assertIs(actual_copy.__func__, actual.__func__)
        self.assertIs(expected_copy.__func__, expected.__func__)

        for copier in (copy.copy, copy.deepcopy):
            self.assert_error_matches(
                lambda copier=copier: copier(actual_module),
                lambda copier=copier: copier(expected_module),
            )
        self.assert_error_matches(
            lambda: copy.deepcopy(actual),
            lambda: copy.deepcopy(expected),
        )

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assert_error_matches(
                    lambda protocol=protocol: pickle.dumps(actual, protocol),
                    lambda protocol=protocol: pickle.dumps(expected, protocol),
                )
                self.assert_error_matches(
                    lambda protocol=protocol: pickle.dumps(actual_module, protocol),
                    lambda protocol=protocol: pickle.dumps(expected_module, protocol),
                )
                self.assertIs(
                    pickle.loads(pickle.dumps(actual.__func__, protocol)),
                    actual.__func__,
                )
                self.assertIs(
                    pickle.loads(pickle.dumps(expected.__func__, protocol)),
                    expected.__func__,
                )

        self.assert_error_matches(
            lambda: pickle.dumps(actual_module.m.is_available),
            lambda: pickle.dumps(expected_module.m.is_available),
        )

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("backends", namespace)
            self.assertNotIn("mkldnn", namespace)

    def reload_contract(self, root):
        parent = root.backends
        module = parent.mkldnn
        old_method = module.is_available
        old_class_method = old_method.__func__
        old_underlying = module.m
        namespace = module.__dict__

        try:
            reloaded = importlib.reload(module)
            parent_function = module.is_available
            reloaded_method = reloaded.is_available
            return (
                reloaded is module,
                module.__dict__ is namespace,
                parent.mkldnn is module,
                sys.modules[module.__name__] is module,
                sys.modules[module.__name__] is reloaded,
                reloaded.m is module,
                module.m is old_underlying,
                type(parent_function).__name__,
                type(reloaded_method).__name__,
                reloaded_method.__self__ is reloaded,
                reloaded_method.__func__ is type(reloaded).is_available,
                reloaded_method.__func__ is old_class_method,
                old_method() is root._C._has_mkldnn,
                parent_function() is root._C._has_mkldnn,
                reloaded_method() is root._C._has_mkldnn,
                copy.copy(reloaded_method) == reloaded_method,
                copy.copy(reloaded_method) is reloaded_method,
                self.error_contract(lambda: copy.deepcopy(reloaded_method)),
                self.error_contract(lambda: pickle.dumps(reloaded_method)),
                self.error_contract(lambda: pickle.dumps(parent_function)),
                self.error_contract(lambda: pickle.dumps(old_class_method)),
            )
        finally:
            self.fresh_mkldnn_module(root)

    def test_reload_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_contract(torch),
            self.reload_contract(reference_torch),
        )

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.backends.mkldnn.is_available
        expected = reference_torch.backends.mkldnn.is_available
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

    def test_reference_mkldnn_execution_exposes_the_unsupported_boundary(self):
        if not reference_torch.backends.mkldnn.is_available():
            self.skipTest("requires an MKLDNN-built reference PyTorch")

        dense = reference_torch.arange(1.0, 7.0).reshape(2, 3)
        mkldnn_tensor = dense.to_mkldnn()
        self.assertIs(mkldnn_tensor.is_mkldnn, True)
        self.assertEqual(str(mkldnn_tensor.layout), "torch._mkldnn")
        self.assertTrue(reference_torch.backends.mkldnn.enabled)
        self.assertTrue(reference_torch.equal(mkldnn_tensor.to_dense(), dense))

        self.assertIs(torch.backends.mkldnn.is_available(), False)
        self.assertIs(torch.tensor([1.0]).is_mkldnn, False)
        self.assertFalse(hasattr(torch.Tensor, "to_mkldnn"))
        self.assertFalse(hasattr(torch, "_mkldnn"))
        with self.assertRaises(RuntimeError):
            torch.tensor([1.0], device="mkldnn")

    def test_configuration_and_execution_surface_remains_unsupported(self):
        actual = torch.backends.mkldnn
        expected = reference_torch.backends.mkldnn
        self.assertEqual(
            {name for name in vars(actual) if not name.startswith("_")},
            {"m"},
        )
        self.assertEqual(
            {name for name in vars(actual) if not name.startswith("_")},
            {name for name in vars(expected) if not name.startswith("_")},
        )
        for name in (
            "VERBOSE_OFF",
            "VERBOSE_ON",
            "VERBOSE_ON_CREATION",
            "allow_tf32",
            "conv",
            "deterministic",
            "enabled",
            "flags",
            "fp32_precision",
            "is_acl_available",
            "matmul",
            "rnn",
            "set_flags",
            "verbose",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(actual, name))
                self.assertTrue(hasattr(expected, name))

        self.assertFalse(hasattr(torch.Tensor, "to_mkldnn"))
        self.assertTrue(hasattr(reference_torch.Tensor, "to_mkldnn"))
        self.assertFalse(hasattr(torch, "_mkldnn"))
        self.assertTrue(hasattr(reference_torch, "_mkldnn"))


if __name__ == "__main__":
    unittest.main()
