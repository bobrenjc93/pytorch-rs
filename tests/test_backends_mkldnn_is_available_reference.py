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
class MkldnnIsAvailableReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "backends.mkldnn.is_available differentials require pinned "
                "PyTorch 2.13.0"
            )

    def normalize(self, value):
        return re.sub(r"0x[0-9a-fA-F]+", "0x...", str(value)).replace(
            "torch_rs",
            "torch",
        )

    def error_contract(self, call):
        try:
            call()
        except Exception as error:
            normalized_args = tuple(
                self.normalize(argument)
                if isinstance(argument, str)
                else argument
                for argument in error.args
            )
            return type(error).__name__, self.normalize(error), normalized_args
        self.fail("the operation unexpectedly succeeded")

    def pickle_shape(self, value, protocol):
        shape = []
        for opcode, argument, _ in pickletools.genops(
            pickle.dumps(value, protocol=protocol)
        ):
            if opcode.name == "FRAME":
                argument = "<frame length>"
            elif isinstance(argument, str):
                argument = self.normalize(argument)
            shape.append((opcode.name, argument))
        return shape

    def fresh_mkldnn_module(self, root):
        module_name = f"{root.__name__}.backends.mkldnn"
        sys.modules.pop(module_name, None)
        if hasattr(root.backends, "mkldnn"):
            del root.backends.mkldnn
        module = importlib.import_module(module_name)
        root.backends.mkldnn = module
        return module

    def test_signature_documentation_and_identity_match_pytorch_2_13(self):
        actual_module = importlib.import_module("torch_rs.backends.mkldnn")
        expected_module = importlib.import_module("torch.backends.mkldnn")
        actual = actual_module.is_available
        expected = expected_module.is_available
        actual_function = actual_module.m.is_available
        expected_function = expected_module.m.is_available

        self.assertIs(torch.backends.mkldnn, actual_module)
        self.assertIs(reference_torch.backends.mkldnn, expected_module)
        self.assertIs(sys.modules[actual_module.__name__], actual_module)
        self.assertIs(sys.modules[expected_module.__name__], expected_module)
        self.assertEqual(type(actual_module).__name__, type(expected_module).__name__)
        self.assertEqual(
            self.normalize(type(actual_module).__module__),
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

        for actual_value, expected_value in (
            (actual, expected),
            (actual_function, expected_function),
        ):
            self.assertIs(type(actual_value), type(expected_value))
            self.assertEqual(
                str(inspect.signature(actual_value)),
                str(inspect.signature(expected_value)),
            )
            self.assertEqual(
                inspect.get_annotations(actual_value),
                inspect.get_annotations(expected_value),
            )
            self.assertEqual(
                typing.get_type_hints(actual_value),
                typing.get_type_hints(expected_value),
            )
            self.assertEqual(actual_value.__name__, expected_value.__name__)
            self.assertEqual(actual_value.__qualname__, expected_value.__qualname__)
            self.assertEqual(
                self.normalize(actual_value.__module__),
                expected_value.__module__,
            )
            self.assertEqual(actual_value.__doc__, expected_value.__doc__)
            self.assertEqual(actual_value.__defaults__, expected_value.__defaults__)
            self.assertEqual(actual_value.__kwdefaults__, expected_value.__kwdefaults__)
            self.assertEqual(actual_value.__dict__, expected_value.__dict__)
            self.assertEqual(
                hasattr(actual_value, "__text_signature__"),
                hasattr(expected_value, "__text_signature__"),
            )
            self.assertEqual(
                actual_value.__code__.co_names,
                expected_value.__code__.co_names,
            )
            self.assertEqual(
                actual_value.__code__.co_freevars,
                expected_value.__code__.co_freevars,
            )
            self.assertEqual(
                actual_value.__code__.co_cellvars,
                expected_value.__code__.co_cellvars,
            )

        self.assertIs(actual.__self__, actual_module)
        self.assertIs(expected.__self__, expected_module)
        self.assertIs(actual.__func__, type(actual_module).is_available)
        self.assertIs(expected.__func__, type(expected_module).is_available)
        self.assertIs(inspect.getmodule(actual), actual_module)
        self.assertIs(inspect.getmodule(expected), expected_module)
        self.assertIs(inspect.getmodule(actual_function), actual_module)
        self.assertIs(inspect.getmodule(expected_function), expected_module)

    def test_imports_copying_and_pickling_match_pytorch_2_13(self):
        actual_backends = importlib.import_module("torch_rs.backends")
        expected_backends = importlib.import_module("torch.backends")
        actual_module = actual_backends.mkldnn
        expected_module = expected_backends.mkldnn
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
                if name
                in {
                    "cuda",
                    "cudnn",
                    "mkl",
                    "mkldnn",
                    "nnpack",
                    "openmp",
                }
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

        for module, method in (
            (actual_module, actual),
            (expected_module, expected),
        ):
            copied = copy.copy(method)
            self.assertIsNot(copied, method)
            self.assertEqual(copied, method)
            self.assertIs(copied.__self__, module)
            self.assertIs(copied.__func__, method.__func__)

        self.assertEqual(
            self.error_contract(lambda: copy.deepcopy(actual)),
            self.error_contract(lambda: copy.deepcopy(expected)),
        )
        for copier in (copy.copy, copy.deepcopy):
            self.assertEqual(
                self.error_contract(lambda: copier(actual_module)),
                self.error_contract(lambda: copier(expected_module)),
            )

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertEqual(
                    self.error_contract(lambda: pickle.dumps(actual, protocol)),
                    self.error_contract(lambda: pickle.dumps(expected, protocol)),
                )
                self.assertEqual(
                    self.error_contract(
                        lambda: pickle.dumps(actual_module.m.is_available, protocol)
                    ),
                    self.error_contract(
                        lambda: pickle.dumps(expected_module.m.is_available, protocol)
                    ),
                )
                self.assertEqual(
                    self.pickle_shape(type(actual_module).is_available, protocol),
                    self.pickle_shape(type(expected_module).is_available, protocol),
                )

    def reload_contract(self, root):
        parent = root.backends
        module = parent.mkldnn
        inner = module.m
        old_method = module.is_available
        old_class_function = type(module).is_available
        namespace = module.__dict__
        reloaded = importlib.reload(module)
        current_method = reloaded.is_available

        contract = (
            reloaded is not module,
            module.__dict__ is namespace,
            parent.mkldnn is module,
            sys.modules[module.__name__] is reloaded,
            reloaded.m is module,
            module.m is inner,
            type(module.is_available).__name__,
            module.is_available is namespace["is_available"],
            type(current_method).__name__,
            current_method.__self__ is reloaded,
            current_method.__func__ is type(reloaded).is_available,
            current_method() is root._C._has_mkldnn,
            copy.copy(old_method) is module.is_available,
            copy.copy(current_method) == current_method,
            copy.copy(current_method) is not current_method,
            self.error_contract(lambda: copy.deepcopy(current_method)),
            self.error_contract(lambda: pickle.dumps(current_method)),
            self.error_contract(lambda: pickle.dumps(module.is_available)),
            self.error_contract(lambda: pickle.dumps(old_class_function)),
        )
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

    def test_build_configuration_and_execution_boundary(self):
        self.assertIs(torch._C._has_mkldnn, False)
        self.assertIs(torch.backends.mkldnn.is_available(), False)
        self.assertFalse(hasattr(torch, "_has_mkldnn"))
        self.assertFalse(hasattr(torch.Tensor, "to_mkldnn"))

        self.assertIs(type(reference_torch._C._has_mkldnn), bool)
        self.assertIs(
            reference_torch.backends.mkldnn.is_available(),
            reference_torch._C._has_mkldnn,
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
                self.assertFalse(hasattr(torch.backends.mkldnn, name))
                self.assertTrue(hasattr(reference_torch.backends.mkldnn, name))


if __name__ == "__main__":
    unittest.main()
