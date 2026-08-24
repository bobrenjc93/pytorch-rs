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

    def error_contract(self, call):
        try:
            call()
        except Exception as error:
            return (
                type(error).__name__,
                self.normalize(error),
                tuple(self.normalize(argument) for argument in error.args),
            )
        self.fail("an operation unexpectedly succeeded")

    def fresh_mkldnn_module(self, root):
        module_name = f"{root.__name__}.backends.mkldnn"
        sys.modules.pop(module_name, None)
        if hasattr(root.backends, "mkldnn"):
            del root.backends.mkldnn
        module = importlib.import_module(module_name)
        root.backends.mkldnn = module
        return module

    def test_native_flag_value_and_placement_match_pytorch_2_13(self):
        actual_flag = torch._C._has_mkldnn
        expected_flag = reference_torch._C._has_mkldnn
        self.assertIs(type(actual_flag), bool)
        self.assertIs(type(expected_flag), bool)
        self.assertIs(actual_flag, False)
        self.assertIs(torch.backends.mkldnn.is_available(), actual_flag)
        self.assertIs(
            reference_torch.backends.mkldnn.is_available(),
            expected_flag,
        )
        for root in (torch, reference_torch):
            self.assertFalse(hasattr(root, "_has_mkldnn"))
            self.assertNotIn("_has_mkldnn", root.__all__)
        self.assertNotIn("_has_mkldnn", torch._C.__all__)

    def test_signature_documentation_and_identity_match_pytorch_2_13(self):
        actual_module = importlib.import_module("torch_rs.backends.mkldnn")
        expected_module = importlib.import_module("torch.backends.mkldnn")
        actual = actual_module.is_available
        expected = expected_module.is_available
        actual_implementation = actual_module.m.is_available
        expected_implementation = expected_module.m.is_available

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

        for method, module in (
            (actual, actual_module),
            (expected, expected_module),
        ):
            self.assertIs(type(method), types.MethodType)
            self.assertIs(method.__self__, module)
            self.assertIs(method.__func__, type(module).is_available)
            self.assertIsNot(method, module.is_available)
            self.assertEqual(method, module.is_available)
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
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )
        self.assertEqual(actual.__code__.co_names, expected.__code__.co_names)
        self.assertEqual(actual.__code__.co_freevars, expected.__code__.co_freevars)
        self.assertEqual(actual.__code__.co_cellvars, expected.__code__.co_cellvars)

        self.assertIs(type(actual_implementation), types.FunctionType)
        self.assertIs(type(expected_implementation), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual_implementation)),
            str(inspect.signature(expected_implementation)),
        )
        self.assertEqual(
            inspect.get_annotations(actual_implementation),
            inspect.get_annotations(expected_implementation),
        )
        self.assertEqual(
            actual_implementation.__qualname__,
            expected_implementation.__qualname__,
        )
        self.assertEqual(
            actual_implementation.__module__.replace("torch_rs", "torch"),
            expected_implementation.__module__,
        )
        self.assertEqual(
            actual_implementation.__doc__,
            expected_implementation.__doc__,
        )
        self.assertEqual(
            actual_implementation.__code__.co_names,
            expected_implementation.__code__.co_names,
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
            imported = method_import["is_available"]
            self.assertIsNot(imported, method)
            self.assertEqual(imported, method)
            self.assertIs(imported.__self__, module)
            self.assertIs(imported.__func__, method.__func__)

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

        for method in (actual, expected):
            copied = copy.copy(method)
            self.assertIsNot(copied, method)
            self.assertEqual(copied, method)
            self.assertIs(copied.__self__, method.__self__)
            self.assertIs(copied.__func__, method.__func__)

        for copier, actual_value, expected_value in (
            (copy.copy, actual_module, expected_module),
            (copy.deepcopy, actual_module, expected_module),
            (copy.deepcopy, actual, expected),
        ):
            self.assertEqual(
                self.error_contract(lambda: copier(actual_value)),
                self.error_contract(lambda: copier(expected_value)),
            )

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                for actual_value, expected_value in (
                    (actual_module, expected_module),
                    (actual, expected),
                    (actual_module.m.is_available, expected_module.m.is_available),
                ):
                    self.assertEqual(
                        self.error_contract(
                            lambda value=actual_value: pickle.dumps(value, protocol)
                        ),
                        self.error_contract(
                            lambda value=expected_value: pickle.dumps(value, protocol)
                        ),
                    )

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("backends", namespace)
            self.assertNotIn("mkldnn", namespace)

        self.assertIs(torch.backends, actual_backends)
        self.assertIs(reference_torch.backends, expected_backends)

    def reload_contract(self, root):
        parent = root.backends
        module = parent.mkldnn
        old_method = module.is_available
        old_implementation = module.m.is_available
        old_underlying = module.m
        namespace = module.__dict__

        try:
            reloaded = importlib.reload(module)
            current_function = module.is_available
            current_method = reloaded.is_available
            copied_method = copy.copy(current_method)
            return (
                reloaded is module,
                module.__dict__ is namespace,
                parent.mkldnn is module,
                sys.modules[module.__name__] is module,
                sys.modules[module.__name__] is reloaded,
                reloaded.m is module,
                module.m is old_underlying,
                type(current_function).__name__,
                type(current_method).__name__,
                current_method.__self__ is reloaded,
                current_method.__func__.__globals__ is module.__dict__,
                old_method() is getattr(root._C, "_has_mkldnn"),
                current_function() is getattr(root._C, "_has_mkldnn"),
                current_method() is getattr(root._C, "_has_mkldnn"),
                old_implementation is current_function,
                copied_method is current_method,
                copied_method == current_method,
                copied_method.__self__ is current_method.__self__,
                self.error_contract(lambda: copy.deepcopy(current_method)),
                self.error_contract(lambda: pickle.dumps(current_function)),
                self.error_contract(lambda: pickle.dumps(current_method)),
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
                self.assertEqual(
                    self.error_contract(actual_call),
                    self.error_contract(expected_call),
                )

    def test_reference_mkldnn_tensors_bound_the_unsupported_execution_surface(self):
        self.assertIs(torch._C._has_mkldnn, False)
        self.assertIs(torch.backends.mkldnn.is_available(), False)
        self.assertIs(
            reference_torch.backends.mkldnn.is_available(),
            reference_torch._C._has_mkldnn,
        )

        self.assertFalse(hasattr(torch, "_mkldnn"))
        self.assertTrue(hasattr(reference_torch, "_mkldnn"))
        self.assertFalse(hasattr(torch.Tensor, "to_mkldnn"))
        self.assertTrue(hasattr(reference_torch.Tensor, "to_mkldnn"))
        self.assertIs(torch.tensor([1.0]).is_mkldnn, False)

        if not reference_torch.backends.mkldnn.is_available():
            self.skipTest("requires an MKLDNN-built reference PyTorch")

        dense = reference_torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mkldnn = dense.to_mkldnn()
        self.assertIs(mkldnn.is_mkldnn, True)
        self.assertEqual(str(mkldnn.layout), "torch._mkldnn")
        self.assertEqual(mkldnn.to_dense().tolist(), dense.tolist())

    def test_configuration_and_verbosity_surface_remains_unsupported(self):
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


if __name__ == "__main__":
    unittest.main()
