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
        return re.sub(
            r"0x[0-9a-fA-F]+",
            "0x...",
            str(value).replace("torch_rs", "torch"),
        )

    def error_contract(self, action):
        try:
            action()
        except Exception as error:
            return (
                type(error).__name__,
                self.normalize(error),
                tuple(self.normalize(argument) for argument in error.args),
            )
        self.fail("the operation unexpectedly succeeded")

    def assert_error_matches(self, actual_call, expected_call):
        self.assertEqual(
            self.error_contract(actual_call),
            self.error_contract(expected_call),
        )

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
            method.__module__.replace("torch_rs", "torch"),
            method.__doc__,
            method.__defaults__,
            method.__kwdefaults__,
            method.__dict__,
            hasattr(method, "__text_signature__"),
            method.__code__.co_names,
            method.__code__.co_freevars,
            method.__code__.co_cellvars,
            method.__code__.co_argcount,
        )

    def function_metadata(self, function):
        return (
            type(function).__name__,
            str(inspect.signature(function)),
            inspect.get_annotations(function),
            typing.get_type_hints(function),
            function.__name__,
            function.__qualname__,
            function.__module__.replace("torch_rs", "torch"),
            function.__doc__,
            function.__defaults__,
            function.__kwdefaults__,
            function.__dict__,
            hasattr(function, "__text_signature__"),
            function.__code__.co_names,
            function.__code__.co_freevars,
            function.__code__.co_cellvars,
            function.__code__.co_argcount,
        )

    def test_value_is_an_exact_build_specific_private_native_flag(self):
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
        self.assertNotIn("_has_mkldnn", torch.__all__)
        self.assertNotIn("_has_mkldnn", torch._C.__all__)

    def test_signature_documentation_and_proxy_identity_match_pytorch_2_13(self):
        actual_module = importlib.import_module("torch_rs.backends.mkldnn")
        expected_module = importlib.import_module("torch.backends.mkldnn")
        actual = actual_module.is_available
        expected = expected_module.is_available
        actual_native = actual_module.m.is_available
        expected_native = expected_module.m.is_available

        self.assertIs(torch.backends.mkldnn, actual_module)
        self.assertIs(reference_torch.backends.mkldnn, expected_module)
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
        self.assertIsNot(actual_module.m, actual_module)
        self.assertIsNot(expected_module.m, expected_module)
        self.assertEqual(
            actual_module.m.__name__,
            expected_module.m.__name__.replace("torch", "torch_rs", 1),
        )
        self.assertIs(actual_module.torch, torch)
        self.assertIs(expected_module.torch, reference_torch)

        self.assertIs(type(actual), types.MethodType)
        self.assertIs(type(expected), types.MethodType)
        self.assertIs(actual.__self__, actual_module)
        self.assertIs(expected.__self__, expected_module)
        self.assertIs(
            actual.__func__,
            type(actual_module).__dict__["is_available"],
        )
        self.assertIs(
            expected.__func__,
            type(expected_module).__dict__["is_available"],
        )
        self.assertIsNot(actual, actual_module.is_available)
        self.assertIsNot(expected, expected_module.is_available)
        self.assertEqual(actual, actual_module.is_available)
        self.assertEqual(expected, expected_module.is_available)
        self.assertEqual(self.method_metadata(actual), self.method_metadata(expected))
        self.assertIs(inspect.getmodule(actual), actual_module)
        self.assertIs(inspect.getmodule(expected), expected_module)

        self.assertIs(type(actual_native), types.FunctionType)
        self.assertIs(type(expected_native), types.FunctionType)
        self.assertEqual(
            self.function_metadata(actual_native),
            self.function_metadata(expected_native),
        )
        self.assertIs(inspect.getmodule(actual_native), actual_module)
        self.assertIs(inspect.getmodule(expected_native), expected_module)

    def test_imports_wildcards_copying_and_pickling_match_pytorch_2_13(self):
        actual_backends = importlib.import_module("torch_rs.backends")
        expected_backends = importlib.import_module("torch.backends")
        actual_module = importlib.import_module("torch_rs.backends.mkldnn")
        expected_module = importlib.import_module("torch.backends.mkldnn")
        actual = actual_module.is_available
        expected = expected_module.is_available
        actual_native = actual_module.m.is_available
        expected_native = expected_module.m.is_available

        self.assertIs(torch.backends, actual_backends)
        self.assertIs(reference_torch.backends, expected_backends)
        self.assertIs(actual_backends.mkldnn, actual_module)
        self.assertIs(expected_backends.mkldnn, expected_module)

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
            imported = function_import["is_available"]
            self.assertIs(backend_import["mkldnn"], module)
            self.assertIsNot(imported, function)
            self.assertEqual(imported, function)
            self.assertIs(imported.__self__, module)
            self.assertIs(imported.__func__, function.__func__)

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
                if name
                in {"cuda", "cudnn", "mkl", "mkldnn", "nnpack", "openmp"}
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
        self.assertIs(actual_child_wildcard["m"], actual_module.m)
        self.assertIs(expected_child_wildcard["m"], expected_module.m)

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("backends", namespace)
            self.assertNotIn("mkldnn", namespace)

        for function, module in (
            (actual, actual_module),
            (expected, expected_module),
        ):
            copied = copy.copy(function)
            self.assertIsNot(copied, function)
            self.assertEqual(copied, function)
            self.assertIs(copied.__self__, module)
            self.assertIs(copied.__func__, function.__func__)

        self.assertEqual(
            self.error_contract(lambda: copy.deepcopy(actual)),
            self.error_contract(lambda: copy.deepcopy(expected)),
        )
        for copier in (copy.copy, copy.deepcopy):
            self.assertEqual(
                self.error_contract(lambda: copier(actual_module)),
                self.error_contract(lambda: copier(expected_module)),
            )

        self.assertIs(copy.copy(actual_native), actual_native)
        self.assertIs(copy.copy(expected_native), expected_native)
        self.assertIs(copy.deepcopy(actual_native), actual_native)
        self.assertIs(copy.deepcopy(expected_native), expected_native)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol, callable="bound method"):
                self.assertEqual(
                    self.error_contract(
                        lambda: pickle.dumps(actual, protocol=protocol)
                    ),
                    self.error_contract(
                        lambda: pickle.dumps(expected, protocol=protocol)
                    ),
                )
            with self.subTest(protocol=protocol, callable="native function"):
                self.assertEqual(
                    self.error_contract(
                        lambda: pickle.dumps(actual_native, protocol=protocol)
                    ),
                    self.error_contract(
                        lambda: pickle.dumps(expected_native, protocol=protocol)
                    ),
                )

    def reload_contract(self, root):
        module = self.fresh_mkldnn_module(root)
        parent = root.backends
        inner = module.m
        namespace = module.__dict__
        old_method = module.is_available
        old_native = inner.is_available

        try:
            reloaded = importlib.reload(module)
            new_method = reloaded.is_available
            new_native = module.is_available
            copied_method = copy.copy(new_method)
            contract = (
                reloaded is module,
                module.__dict__ is namespace,
                parent.mkldnn is module,
                sys.modules[module.__name__] is module,
                sys.modules[module.__name__] is reloaded,
                reloaded.m is module,
                module.m is inner,
                type(old_method).__name__,
                old_method.__self__ is module,
                type(new_method).__name__,
                new_method.__self__ is reloaded,
                type(new_native).__name__,
                reloaded.m.is_available is new_native,
                inner.is_available is old_native,
                old_method() is root._C._has_mkldnn,
                old_native() is root._C._has_mkldnn,
                new_native() is root._C._has_mkldnn,
                new_method() is root._C._has_mkldnn,
                copied_method is new_method,
                copied_method == new_method,
                copied_method.__self__ is reloaded,
                copied_method.__func__ is new_method.__func__,
                self.error_contract(lambda: copy.deepcopy(new_method)),
                self.error_contract(lambda: pickle.dumps(new_method)),
                self.error_contract(lambda: pickle.dumps(old_native)),
                self.error_contract(lambda: pickle.dumps(new_native)),
            )
        finally:
            self.fresh_mkldnn_module(root)
        return contract

    def test_reload_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_contract(torch),
            self.reload_contract(reference_torch),
        )

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.backends.mkldnn.is_available
        expected = reference_torch.backends.mkldnn.is_available
        actual_native = torch.backends.mkldnn.m.is_available
        expected_native = reference_torch.backends.mkldnn.m.is_available

        public_cases = (
            (lambda: actual(None), lambda: expected(None)),
            (lambda: actual(None, None), lambda: expected(None, None)),
            (lambda: actual(enabled=True), lambda: expected(enabled=True)),
            (
                lambda: actual(None, enabled=True),
                lambda: expected(None, enabled=True),
            ),
        )
        native_cases = (
            (lambda: actual_native(None), lambda: expected_native(None)),
            (
                lambda: actual_native(None, None),
                lambda: expected_native(None, None),
            ),
            (
                lambda: actual_native(enabled=True),
                lambda: expected_native(enabled=True),
            ),
            (
                lambda: actual_native(None, enabled=True),
                lambda: expected_native(None, enabled=True),
            ),
        )
        for kind, cases in (("public", public_cases), ("native", native_cases)):
            for case, (actual_call, expected_call) in enumerate(cases):
                with self.subTest(kind=kind, case=case):
                    self.assert_error_matches(actual_call, expected_call)

    def test_mkldnn_configuration_tensors_and_execution_remain_unsupported(self):
        actual = torch.backends.mkldnn
        expected = reference_torch.backends.mkldnn
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

        for name in (
            "_get_mkldnn_enabled",
            "_set_mkldnn_enabled",
            "_get_mkldnn_deterministic",
            "_set_mkldnn_deterministic",
            "_get_onednn_allow_tf32",
            "_set_onednn_allow_tf32",
            "_has_mkldnn_acl",
            "_verbose",
        ):
            with self.subTest(native_name=name):
                self.assertFalse(hasattr(torch._C, name))
                self.assertTrue(hasattr(reference_torch._C, name))

        actual_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        expected_tensor = reference_torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.assertIs(actual_tensor.is_mkldnn, False)
        self.assertIs(expected_tensor.is_mkldnn, False)
        self.assertFalse(hasattr(torch.Tensor, "to_mkldnn"))
        self.assertTrue(hasattr(reference_torch.Tensor, "to_mkldnn"))
        self.assertFalse(hasattr(torch, "_mkldnn"))
        self.assertTrue(hasattr(reference_torch, "_mkldnn"))

        if not expected.is_available():
            self.skipTest("PyTorch oneDNN backend is unavailable")

        mkldnn_tensor = expected_tensor.to_mkldnn()
        result = (mkldnn_tensor + mkldnn_tensor).to_dense()
        self.assertIs(mkldnn_tensor.is_mkldnn, True)
        self.assertEqual(str(mkldnn_tensor.layout), "torch._mkldnn")
        self.assertEqual(result.tolist(), [[2.0, 4.0], [6.0, 8.0]])
        self.assertIs(actual.is_available(), False)


if __name__ == "__main__":
    unittest.main()
