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


BACKENDS = {
    "openmp": "has_openmp",
    "mkl": "has_mkl",
    "nnpack": None,
}


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class NativeCpuBackendAvailabilityReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "native CPU backend availability differentials require pinned "
                "PyTorch 2.13.0"
            )

    def normalize(self, value):
        return str(value).replace("torch_rs", "torch")

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

    def test_values_are_exact_build_specific_native_flags(self):
        for backend, flag in BACKENDS.items():
            with self.subTest(backend=backend):
                actual = getattr(torch.backends, backend).is_available()
                expected = getattr(reference_torch.backends, backend).is_available()
                self.assertIs(type(actual), bool)
                self.assertIs(type(expected), bool)
                self.assertIs(actual, False)
                if flag is None:
                    self.assertIs(actual, torch._nnpack_available())
                    self.assertIs(expected, reference_torch._nnpack_available())
                else:
                    self.assertIs(actual, getattr(torch._C, flag))
                    self.assertIs(actual, getattr(torch, flag))
                    self.assertIs(expected, getattr(reference_torch._C, flag))

    def test_signature_documentation_and_ownership_match_pytorch_2_13(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                actual_module = importlib.import_module(
                    f"torch_rs.backends.{backend}"
                )
                expected_module = importlib.import_module(
                    f"torch.backends.{backend}"
                )
                actual = actual_module.is_available
                expected = expected_module.is_available

                self.assertIsNone(actual_module.__doc__)
                self.assertEqual(actual_module.__doc__, expected_module.__doc__)
                self.assertEqual(
                    hasattr(actual_module, "__all__"),
                    hasattr(expected_module, "__all__"),
                )
                self.assertIs(type(actual), types.FunctionType)
                self.assertIs(type(expected), types.FunctionType)
                self.assertEqual(
                    str(inspect.signature(actual)),
                    str(inspect.signature(expected)),
                )
                self.assertEqual(
                    inspect.get_annotations(actual),
                    inspect.get_annotations(expected),
                )
                self.assertEqual(
                    typing.get_type_hints(actual),
                    typing.get_type_hints(expected),
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
                self.assertEqual(
                    actual.__code__.co_names,
                    expected.__code__.co_names,
                )

    def test_imports_wildcards_copying_and_pickling_match_supported_scope(self):
        actual_backends = importlib.import_module("torch_rs.backends")
        expected_backends = importlib.import_module("torch.backends")
        self.assertIs(torch.backends, actual_backends)
        self.assertIs(reference_torch.backends, expected_backends)
        self.assertIs(sys.modules["torch_rs.backends"], actual_backends)
        self.assertIs(sys.modules["torch.backends"], expected_backends)
        self.assertEqual(actual_backends.__doc__, expected_backends.__doc__)
        self.assertEqual(
            hasattr(actual_backends, "__all__"),
            hasattr(expected_backends, "__all__"),
        )
        self.assertEqual(
            torch.__all__.count("backends"),
            reference_torch.__all__.count("backends"),
        )

        actual_package_import = {}
        expected_package_import = {}
        exec("from torch_rs import backends", actual_package_import)
        exec("from torch import backends", expected_package_import)
        self.assertIs(actual_package_import["backends"], actual_backends)
        self.assertIs(expected_package_import["backends"], expected_backends)

        actual_parent_wildcard = {}
        expected_parent_wildcard = {}
        exec("from torch_rs.backends import *", actual_parent_wildcard)
        exec("from torch.backends import *", expected_parent_wildcard)
        actual_parent_names = {
            name
            for name in actual_parent_wildcard
            if not name.startswith("__")
        }
        expected_supported_names = {
            name
            for name in expected_parent_wildcard
            if name
            in {*BACKENDS, "cpu", "cuda", "cudnn", "kleidiai", "m", "mha"}
        }
        self.assertEqual(actual_parent_names, expected_supported_names)

        for backend in BACKENDS:
            with self.subTest(backend=backend):
                actual_module = getattr(actual_backends, backend)
                expected_module = getattr(expected_backends, backend)
                actual = actual_module.is_available
                expected = expected_module.is_available
                self.assertIs(
                    importlib.import_module(f"torch_rs.backends.{backend}"),
                    actual_module,
                )
                self.assertIs(
                    importlib.import_module(f"torch.backends.{backend}"),
                    expected_module,
                )

                actual_child_wildcard = {}
                expected_child_wildcard = {}
                exec(
                    f"from torch_rs.backends.{backend} import *",
                    actual_child_wildcard,
                )
                exec(
                    f"from torch.backends.{backend} import *",
                    expected_child_wildcard,
                )
                actual_child_names = {
                    name
                    for name in actual_child_wildcard
                    if not name.startswith("__")
                }
                expected_supported_child_names = {
                    name
                    for name in expected_child_wildcard
                    if name in {"flags", "is_available", "set_flags", "torch"}
                }
                self.assertEqual(
                    actual_child_names,
                    expected_supported_child_names,
                )

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

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("backends", namespace)

    def reload_contract(self, root, backend):
        parent = root.backends
        module = getattr(parent, backend)
        old_function = module.is_available
        namespace = module.__dict__
        reloaded = importlib.reload(module)
        new_function = module.is_available

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
            self.fail("a stale backend availability function remained pickleable")

        return (
            reloaded is module,
            module.__dict__ is namespace,
            getattr(parent, backend) is module,
            sys.modules[module.__name__] is module,
            old_function is not new_function,
            copy.copy(new_function) is new_function,
            copy.deepcopy(new_function) is new_function,
            pickle.loads(pickle.dumps(new_function)) is new_function,
            stale_pickle_error,
        )

    def test_reload_behavior_matches_pytorch_2_13(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                self.assertEqual(
                    self.reload_contract(torch, backend),
                    self.reload_contract(reference_torch, backend),
                )
                actual = getattr(torch.backends, backend).is_available
                expected = getattr(reference_torch.backends, backend).is_available
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    self.assertEqual(
                        self.pickle_shape(actual, protocol),
                        self.pickle_shape(expected, protocol),
                    )

    def test_argument_errors_match_pytorch_2_13(self):
        for backend in BACKENDS:
            actual = getattr(torch.backends, backend).is_available
            expected = getattr(reference_torch.backends, backend).is_available
            cases = (
                (lambda: actual(None), lambda: expected(None)),
                (lambda: actual(None, None), lambda: expected(None, None)),
                (
                    lambda: actual(enabled=True),
                    lambda: expected(enabled=True),
                ),
                (
                    lambda: actual(None, enabled=True),
                    lambda: expected(None, enabled=True),
                ),
            )
            for case, (actual_call, expected_call) in enumerate(cases):
                with self.subTest(backend=backend, case=case):
                    self.assert_error_matches(actual_call, expected_call)

    def test_configuration_verbosity_and_other_backends_remain_unsupported(self):
        actual_backends = torch.backends
        expected_backends = reference_torch.backends
        actual_public = {
            name for name in vars(actual_backends) if not name.startswith("_")
        }
        expected_public = {
            name for name in vars(expected_backends) if not name.startswith("_")
        }
        self.assertEqual(
            actual_public,
            {*BACKENDS, "cpu", "cuda", "cudnn", "kleidiai", "m", "mha"},
        )
        self.assertTrue(set(BACKENDS).issubset(expected_public))
        self.assertTrue(
            {
                "mkldnn",
                "mps",
                "quantized",
            }.issubset(expected_public - actual_public)
        )

        self.assertFalse(hasattr(torch.backends, "flags"))
        self.assertTrue(hasattr(reference_torch.backends, "flags"))
        for name in ("VERBOSE_OFF", "VERBOSE_ON", "verbose"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.backends.mkl, name))
                self.assertTrue(hasattr(reference_torch.backends.mkl, name))

        self.assertEqual(
            torch.backends.nnpack.__all__,
            ["is_available", "flags", "set_flags"],
        )
        self.assertEqual(
            torch.backends.nnpack.__all__,
            reference_torch.backends.nnpack.__all__,
        )


if __name__ == "__main__":
    unittest.main()
