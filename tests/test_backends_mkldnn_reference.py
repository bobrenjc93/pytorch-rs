import copy
import importlib
import inspect
import pickle
import re
import sys
import types
import unittest
import warnings

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


UNSUPPORTED_MKLDNN_NAMES = (
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
)


def fresh_mkldnn_module(root):
    module_name = f"{root.__name__}.backends.mkldnn"
    sys.modules.pop(module_name, None)
    if hasattr(root.backends, "mkldnn"):
        del root.backends.mkldnn
    module = importlib.import_module(module_name)
    root.backends.mkldnn = module
    return module


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class MkldnnAvailabilityReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "backends.mkldnn.is_available differentials require pinned "
                "PyTorch 2.13.0"
            )

    def setUp(self):
        self.actual = fresh_mkldnn_module(torch)
        self.expected = fresh_mkldnn_module(reference_torch)

    def tearDown(self):
        fresh_mkldnn_module(torch)
        fresh_mkldnn_module(reference_torch)

    def normalize(self, value):
        if isinstance(value, str):
            return value.replace("torch_rs", "torch")
        return value

    def method_contract(self, module):
        function = module.is_available
        return {
            "type": type(function).__name__,
            "signature": str(inspect.signature(function)),
            "annotations": inspect.get_annotations(function),
            "name": function.__name__,
            "qualname": function.__qualname__,
            "module": self.normalize(function.__module__),
            "doc": function.__doc__,
            "defaults": function.__defaults__,
            "kwdefaults": function.__kwdefaults__,
            "dict": function.__dict__,
            "has_text_signature": hasattr(function, "__text_signature__"),
            "self_is_module": function.__self__ is module,
            "code_names": function.__func__.__code__.co_names,
            "code_freevars": function.__func__.__code__.co_freevars,
            "code_cellvars": function.__func__.__code__.co_cellvars,
        }

    def test_available_probe_matches_pytorch_2_13_supported_subset(self):
        actual = self.actual.is_available()
        expected = self.expected.is_available()

        self.assertIs(type(actual), bool)
        self.assertIs(type(expected), bool)
        self.assertIs(actual, False)
        self.assertIs(actual, torch._C._has_mkldnn)
        self.assertIs(expected, reference_torch._C._has_mkldnn)

    def test_signature_and_module_shape_match_pytorch_2_13(self):
        actual = self.actual
        expected = self.expected

        self.assertEqual(type(actual).__name__, type(expected).__name__)
        self.assertEqual(
            self.normalize(type(actual).__module__),
            type(expected).__module__,
        )
        self.assertIsInstance(actual, types.ModuleType)
        self.assertIsInstance(expected, types.ModuleType)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(hasattr(actual, "__all__"), hasattr(expected, "__all__"))
        self.assertEqual(
            {name for name in vars(actual) if not name.startswith("_")},
            {name for name in vars(expected) if not name.startswith("_")},
        )
        self.assertEqual(
            {name for name in vars(actual) if not name.startswith("_")},
            {"m"},
        )
        self.assertIs(type(actual.m), types.ModuleType)
        self.assertIs(type(expected.m), types.ModuleType)
        self.assertIsNot(actual.m, actual)
        self.assertIsNot(expected.m, expected)
        self.assertEqual(
            actual.m.__name__.replace("torch_rs", "torch"),
            expected.m.__name__,
        )
        self.assertEqual(self.method_contract(actual), self.method_contract(expected))

        actual_impl = actual.m.is_available
        expected_impl = expected.m.is_available
        self.assertIs(type(actual_impl), types.FunctionType)
        self.assertIs(type(expected_impl), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual_impl)),
            str(inspect.signature(expected_impl)),
        )
        self.assertEqual(
            inspect.get_annotations(actual_impl),
            inspect.get_annotations(expected_impl),
        )
        self.assertEqual(actual_impl.__name__, expected_impl.__name__)
        self.assertEqual(actual_impl.__qualname__, expected_impl.__qualname__)
        self.assertEqual(
            self.normalize(actual_impl.__module__),
            expected_impl.__module__,
        )
        self.assertEqual(actual_impl.__doc__, expected_impl.__doc__)
        self.assertEqual(actual_impl.__defaults__, expected_impl.__defaults__)
        self.assertEqual(actual_impl.__kwdefaults__, expected_impl.__kwdefaults__)
        self.assertEqual(actual_impl.__dict__, expected_impl.__dict__)
        self.assertEqual(
            hasattr(actual_impl, "__text_signature__"),
            hasattr(expected_impl, "__text_signature__"),
        )

    def test_imports_wildcards_copying_and_pickling_match_pytorch_2_13(self):
        for package_name, module in (
            ("torch_rs", self.actual),
            ("torch", self.expected),
        ):
            backend_import = {}
            function_import = {}
            parent_wildcard = {}
            child_wildcard = {}
            exec(f"from {package_name}.backends import mkldnn", backend_import)
            exec(
                f"from {package_name}.backends.mkldnn import is_available",
                function_import,
            )
            exec(f"from {package_name}.backends import *", parent_wildcard)
            exec(f"from {package_name}.backends.mkldnn import *", child_wildcard)

            self.assertIs(backend_import["mkldnn"], module)
            self.assertEqual(function_import["is_available"], module.is_available)
            self.assertIs(function_import["is_available"].__self__, module)
            self.assertIs(parent_wildcard["mkldnn"], module)
            self.assertEqual(
                {name for name in child_wildcard if not name.startswith("__")},
                {"m"},
            )

            copied = copy.copy(module.is_available)
            self.assertIsNot(copied, module.is_available)
            self.assertEqual(copied, module.is_available)
            self.assertIs(copied.__self__, module)
            self.assertIs(copied.__func__, module.is_available.__func__)

            for copier in (copy.copy, copy.deepcopy):
                with self.assertRaisesRegex(
                    TypeError,
                    "^cannot pickle 'MkldnnModule' object$",
                ):
                    copier(module)
            with self.assertRaisesRegex(
                TypeError,
                "^cannot pickle 'MkldnnModule' object$",
            ):
                copy.deepcopy(module.is_available)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(package=package_name, protocol=protocol):
                    with self.assertRaises(TypeError) as raised:
                        pickle.dumps(module.is_available, protocol=protocol)
                    self.assertIn(
                        str(raised.exception),
                        {
                            "module() argument 'name' must be str, not MkldnnModule",
                            "cannot pickle 'MkldnnModule' object",
                        },
                    )

        namespace = {}
        exec("from torch_rs import *", namespace)
        self.assertNotIn("backends", namespace)
        self.assertNotIn("mkldnn", namespace)

    def reload_contract(self, root):
        parent = root.backends
        module = parent.mkldnn
        namespace = module.__dict__
        old_method = module.is_available
        reloaded = importlib.reload(module)

        try:
            pickle.dumps(module.is_available)
        except Exception as error:
            stale_pickle_error = (
                type(error).__name__,
                self.normalize(re.sub(r"0x[0-9a-fA-F]+", "0x...", str(error))),
            )
        else:
            self.fail("a stale MKLDNN availability query remained pickleable")

        return (
            reloaded is not module,
            module.__dict__ is namespace,
            parent.mkldnn is module,
            sys.modules[module.__name__] is reloaded,
            reloaded.m is module,
            type(old_method).__name__,
            type(module.is_available).__name__,
            type(reloaded.is_available).__name__,
            module.is_available() is root._C._has_mkldnn,
            reloaded.is_available() is root._C._has_mkldnn,
            stale_pickle_error,
        )

    def test_reload_behavior_matches_pytorch_2_13(self):
        try:
            self.assertEqual(
                self.reload_contract(torch),
                self.reload_contract(reference_torch),
            )
        finally:
            self.actual = fresh_mkldnn_module(torch)
            self.expected = fresh_mkldnn_module(reference_torch)

    def test_argument_errors_match_pytorch_2_13(self):
        actual = self.actual.is_available
        expected = self.expected.is_available
        cases = (
            ((None,), {}),
            ((None, None), {}),
            ((), {"enabled": True}),
            ((None,), {"enabled": True}),
        )
        for args, kwargs in cases:
            with self.subTest(args=args, kwargs=kwargs):
                with self.assertRaises(Exception) as actual_raised:
                    actual(*args, **kwargs)
                with self.assertRaises(Exception) as expected_raised:
                    expected(*args, **kwargs)
                self.assertIs(
                    type(actual_raised.exception),
                    type(expected_raised.exception),
                )
                self.assertEqual(
                    self.normalize(str(actual_raised.exception)),
                    str(expected_raised.exception),
                )
                self.assertEqual(
                    tuple(
                        self.normalize(argument)
                        for argument in actual_raised.exception.args
                    ),
                    expected_raised.exception.args,
                )

    def test_repo_keeps_mkldnn_flags_and_execution_out_of_scope(self):
        for name in UNSUPPORTED_MKLDNN_NAMES:
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.backends.mkldnn, name))
                self.assertTrue(hasattr(reference_torch.backends.mkldnn, name))

        self.assertFalse(hasattr(torch, "has_mkldnn"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.assertTrue(hasattr(reference_torch, "has_mkldnn"))
        self.assertFalse(hasattr(torch.Tensor, "to_mkldnn"))
        self.assertTrue(hasattr(reference_torch.Tensor, "to_mkldnn"))
        self.assertFalse(hasattr(torch, "_mkldnn"))
        self.assertTrue(hasattr(reference_torch, "_mkldnn"))
        self.assertIs(torch.tensor([1.0]).is_mkldnn, False)
        self.assertIs(torch.tensor([1.0]).layout, torch.strided)


if __name__ == "__main__":
    unittest.main()
