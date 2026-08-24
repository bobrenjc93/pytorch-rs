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

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(
            self.normalize(actual_raised.exception),
            self.normalize(expected_raised.exception),
        )
        self.assertEqual(
            tuple(self.normalize(arg) for arg in actual_raised.exception.args),
            tuple(self.normalize(arg) for arg in expected_raised.exception.args),
        )

    def pickle_shape(self, value, protocol):
        shape = []
        for opcode, argument, _ in pickletools.genops(
            pickle.dumps(value, protocol=protocol)
        ):
            if opcode.name == "FRAME":
                argument = "<frame length>"
            elif isinstance(argument, str):
                argument = argument.replace("torch_rs", "torch")
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

    def test_value_is_an_exact_build_specific_native_flag(self):
        actual_module = torch.backends.mkldnn
        expected_module = reference_torch.backends.mkldnn
        for module, backend in (
            (torch, actual_module),
            (reference_torch, expected_module),
        ):
            self.assertIs(type(module._C._has_mkldnn), bool)
            self.assertIs(backend.is_available(), module._C._has_mkldnn)
            self.assertIs(backend.m.is_available(), module._C._has_mkldnn)

        self.assertIs(torch._C._has_mkldnn, False)
        self.assertFalse(hasattr(torch, "_has_mkldnn"))
        self.assertFalse(hasattr(reference_torch, "_has_mkldnn"))

    def test_signature_documentation_and_identity_match_pytorch_2_13(self):
        actual_module = importlib.import_module("torch_rs.backends.mkldnn")
        expected_module = importlib.import_module("torch.backends.mkldnn")
        actual = actual_module.is_available
        expected = expected_module.is_available
        actual_underlying = actual_module.m.is_available
        expected_underlying = expected_module.m.is_available

        self.assertIs(torch.backends.mkldnn, actual_module)
        self.assertIs(reference_torch.backends.mkldnn, expected_module)
        self.assertIs(sys.modules[actual_module.__name__], actual_module)
        self.assertIs(sys.modules[expected_module.__name__], expected_module)
        self.assertEqual(type(actual_module).__name__, type(expected_module).__name__)
        self.assertEqual(
            type(actual_module).__module__.replace("torch_rs", "torch"),
            type(expected_module).__module__,
        )
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

        for value, expected_value, module in (
            (actual, expected, actual_module),
            (actual_underlying, expected_underlying, actual_module),
        ):
            self.assertIs(type(value), type(expected_value))
            self.assertEqual(
                str(inspect.signature(value)),
                str(inspect.signature(expected_value)),
            )
            self.assertEqual(
                inspect.get_annotations(value),
                inspect.get_annotations(expected_value),
            )
            self.assertEqual(
                typing.get_type_hints(value),
                typing.get_type_hints(expected_value),
            )
            self.assertEqual(value.__name__, expected_value.__name__)
            self.assertEqual(value.__qualname__, expected_value.__qualname__)
            self.assertEqual(
                value.__module__.replace("torch_rs", "torch"),
                expected_value.__module__,
            )
            self.assertIs(inspect.getmodule(value), module)
            self.assertEqual(value.__doc__, expected_value.__doc__)
            self.assertEqual(value.__defaults__, expected_value.__defaults__)
            self.assertEqual(value.__kwdefaults__, expected_value.__kwdefaults__)
            self.assertEqual(value.__dict__, expected_value.__dict__)
            self.assertEqual(
                hasattr(value, "__text_signature__"),
                hasattr(expected_value, "__text_signature__"),
            )
            self.assertEqual(value.__code__.co_names, expected_value.__code__.co_names)
            self.assertEqual(
                value.__code__.co_freevars,
                expected_value.__code__.co_freevars,
            )
            self.assertEqual(
                value.__code__.co_cellvars,
                expected_value.__code__.co_cellvars,
            )

        self.assertIs(actual.__self__, actual_module)
        self.assertIs(expected.__self__, expected_module)
        self.assertIs(actual.__func__, type(actual_module).__dict__["is_available"])
        self.assertIs(expected.__func__, type(expected_module).__dict__["is_available"])

    def test_imports_wildcards_copying_and_pickling_match_pytorch_2_13(self):
        actual_backends = importlib.import_module("torch_rs.backends")
        expected_backends = importlib.import_module("torch.backends")
        actual_module = importlib.import_module("torch_rs.backends.mkldnn")
        expected_module = importlib.import_module("torch.backends.mkldnn")
        actual = actual_module.is_available
        expected = expected_module.is_available

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
            imported = function_import["is_available"]
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
        self.assertEqual(
            {name for name in actual_child_wildcard if not name.startswith("__")},
            {"m"},
        )

        for function, module in (
            (actual, actual_module),
            (expected, expected_module),
        ):
            copied = copy.copy(function)
            self.assertIsNot(copied, function)
            self.assertEqual(copied, function)
            self.assertIs(copied.__self__, module)
            self.assertIs(copied.__func__, function.__func__)

        self.assert_error_matches(
            lambda: copy.deepcopy(actual),
            lambda: copy.deepcopy(expected),
        )
        for copier in (copy.copy, copy.deepcopy):
            self.assert_error_matches(
                lambda copier=copier: copier(actual_module),
                lambda copier=copier: copier(expected_module),
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
                self.assert_error_matches(
                    lambda protocol=protocol: pickle.dumps(
                        actual_module.m.is_available,
                        protocol,
                    ),
                    lambda protocol=protocol: pickle.dumps(
                        expected_module.m.is_available,
                        protocol,
                    ),
                )
                actual_descriptor = type(actual_module).__dict__["is_available"]
                expected_descriptor = type(expected_module).__dict__["is_available"]
                self.assertIs(
                    pickle.loads(pickle.dumps(actual_descriptor, protocol)),
                    actual_descriptor,
                )
                self.assertIs(
                    pickle.loads(pickle.dumps(expected_descriptor, protocol)),
                    expected_descriptor,
                )
                self.assertEqual(
                    self.pickle_shape(actual_descriptor, protocol),
                    self.pickle_shape(expected_descriptor, protocol),
                )

        self.assertIs(torch.backends, actual_backends)
        self.assertIs(reference_torch.backends, expected_backends)

    def reload_contract(self, root):
        parent = root.backends
        module = parent.mkldnn
        old_function = module.is_available
        old_underlying = module.m.is_available
        old_descriptor = type(module).__dict__["is_available"]
        namespace = module.__dict__

        try:
            reloaded = importlib.reload(module)
            new_function = reloaded.is_available

            stale_pickle_errors = []
            for stale in (old_underlying, old_descriptor, module.is_available):
                try:
                    pickle.dumps(stale)
                except Exception as error:
                    stale_pickle_errors.append(
                        (type(error).__name__, self.normalize(error))
                    )
                else:
                    self.fail("a stale MKLDNN callable remained pickleable")

            try:
                copy.deepcopy(new_function)
            except Exception as error:
                deepcopy_error = (type(error).__name__, self.normalize(error))
            else:
                self.fail("a bound MKLDNN query unexpectedly supported deepcopy")

            try:
                pickle.dumps(new_function)
            except Exception as error:
                pickle_error = (type(error).__name__, self.normalize(error))
            else:
                self.fail("a bound MKLDNN query unexpectedly supported pickling")

            copied = copy.copy(new_function)
            return (
                reloaded is module,
                module.__dict__ is namespace,
                parent.mkldnn is module,
                sys.modules[module.__name__] is module,
                sys.modules[module.__name__] is reloaded,
                reloaded.m is module,
                type(module.is_available).__name__,
                type(new_function).__name__,
                new_function.__self__ is reloaded,
                copy.copy(old_function) is module.is_available,
                copied is new_function,
                copied == new_function,
                copied.__self__ is reloaded,
                copied.__func__ is new_function.__func__,
                deepcopy_error,
                pickle_error,
                tuple(stale_pickle_errors),
            )
        finally:
            self.fresh_mkldnn_module(root)

    def test_reload_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_contract(torch),
            self.reload_contract(reference_torch),
        )

    def test_argument_errors_match_pytorch_2_13(self):
        actual_module = torch.backends.mkldnn
        expected_module = reference_torch.backends.mkldnn
        actual = actual_module.is_available
        expected = expected_module.is_available
        cases = (
            (lambda: actual(None), lambda: expected(None)),
            (lambda: actual(None, None), lambda: expected(None, None)),
            (lambda: actual(enabled=True), lambda: expected(enabled=True)),
            (
                lambda: actual(None, enabled=True),
                lambda: expected(None, enabled=True),
            ),
            (lambda: actual(self=None), lambda: expected(self=None)),
            (
                lambda: actual_module.m.is_available(None),
                lambda: expected_module.m.is_available(None),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_reference_execution_is_bounded_by_unsupported_mkldnn_surface(self):
        if not reference_torch.backends.mkldnn.is_available():
            self.skipTest("requires an MKLDNN-built reference PyTorch")

        self.assertIs(reference_torch._C._has_mkldnn, True)
        self.assertIs(reference_torch.backends.mkldnn.is_available(), True)
        self.assertIs(torch._C._has_mkldnn, False)
        self.assertIs(torch.backends.mkldnn.is_available(), False)

        dense = reference_torch.arange(1.0, 5.0).reshape(2, 2)
        mkldnn = dense.to_mkldnn()
        result = mkldnn.relu().to_dense()
        self.assertIs(mkldnn.is_mkldnn, True)
        self.assertEqual(str(mkldnn.layout), "torch._mkldnn")
        self.assertEqual(result.tolist(), [[1.0, 2.0], [3.0, 4.0]])

        self.assertFalse(hasattr(torch.Tensor, "to_mkldnn"))
        self.assertFalse(hasattr(torch, "_mkldnn"))
        self.assertIs(torch.tensor([1.0]).is_mkldnn, False)

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

        self.assertFalse(hasattr(torch._C, "_has_mkldnn_acl"))
        self.assertTrue(hasattr(reference_torch._C, "_has_mkldnn_acl"))
        self.assertFalse(hasattr(torch.Tensor, "to_mkldnn"))
        self.assertTrue(hasattr(reference_torch.Tensor, "to_mkldnn"))


if __name__ == "__main__":
    unittest.main()
