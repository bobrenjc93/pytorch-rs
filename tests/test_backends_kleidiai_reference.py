import copy
import importlib
import inspect
import pickle
import pickletools
import re
import sys
import threading
import types
import typing
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class KleidiAIAvailabilityReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "backends.kleidiai.is_available differentials require pinned "
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

    def test_value_and_private_native_placement_match_pytorch_2_13(self):
        actual = torch.backends.kleidiai.is_available()
        expected = reference_torch.backends.kleidiai.is_available()

        self.assertIs(type(actual), bool)
        self.assertIs(type(expected), bool)
        self.assertIs(actual, torch._C._has_kleidiai)
        self.assertIs(expected, reference_torch._C._has_kleidiai)
        self.assertIs(actual, False)
        self.assertIs(expected, False)

        for root in (torch, reference_torch):
            native_import = {}
            package_wildcard = {}
            native_wildcard = {}
            exec(f"from {root.__name__}._C import _has_kleidiai", native_import)
            exec(f"from {root.__name__} import *", package_wildcard)
            exec(f"from {root.__name__}._C import *", native_wildcard)
            self.assertIs(native_import["_has_kleidiai"], False)
            self.assertFalse(hasattr(root, "_has_kleidiai"))
            self.assertNotIn("_has_kleidiai", root.__all__)
            self.assertNotIn("_has_kleidiai", package_wildcard)
            self.assertNotIn("_has_kleidiai", native_wildcard)

    def test_signature_documentation_and_identity_match_pytorch_2_13(self):
        actual_module = importlib.import_module("torch_rs.backends.kleidiai")
        expected_module = importlib.import_module("torch.backends.kleidiai")
        actual = actual_module.is_available
        expected = expected_module.is_available

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
        self.assertEqual(actual.__code__.co_names, expected.__code__.co_names)
        self.assertEqual(actual.__code__.co_freevars, expected.__code__.co_freevars)
        self.assertEqual(actual.__code__.co_cellvars, expected.__code__.co_cellvars)

    def test_imports_copying_and_pickling_match_pytorch_2_13(self):
        actual_backends = importlib.import_module("torch_rs.backends")
        expected_backends = importlib.import_module("torch.backends")
        actual_module = importlib.import_module("torch_rs.backends.kleidiai")
        expected_module = importlib.import_module("torch.backends.kleidiai")
        actual = actual_module.is_available
        expected = expected_module.is_available

        self.assertIs(torch.backends, actual_backends)
        self.assertIs(reference_torch.backends, expected_backends)
        self.assertIs(actual_backends.kleidiai, actual_module)
        self.assertIs(expected_backends.kleidiai, expected_module)
        self.assertIs(sys.modules[actual_module.__name__], actual_module)
        self.assertIs(sys.modules[expected_module.__name__], expected_module)

        for package_name, module, function in (
            ("torch_rs", actual_module, actual),
            ("torch", expected_module, expected),
        ):
            backend_import = {}
            function_import = {}
            parent_wildcard = {}
            child_wildcard = {}
            exec(f"from {package_name}.backends import kleidiai", backend_import)
            exec(
                f"from {package_name}.backends.kleidiai import is_available",
                function_import,
            )
            exec(f"from {package_name}.backends import *", parent_wildcard)
            exec(f"from {package_name}.backends.kleidiai import *", child_wildcard)
            self.assertIs(backend_import["kleidiai"], module)
            self.assertIs(function_import["is_available"], function)
            self.assertIs(parent_wildcard["kleidiai"], module)
            self.assertEqual(
                {name for name in child_wildcard if not name.startswith("__")},
                {"is_available", "torch"},
            )

        self.assertIs(copy.copy(actual), actual)
        self.assertIs(copy.copy(expected), expected)
        self.assertIs(copy.deepcopy(actual), actual)
        self.assertIs(copy.deepcopy(expected), expected)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(
                    pickle.loads(pickle.dumps(expected, protocol)),
                    expected,
                )
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

    def threaded_contract(self, root):
        function = root.backends.kleidiai.is_available
        flag = root._C._has_kleidiai
        worker_count = 16
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                barrier.wait(timeout=5)
                value = function()
                results[index] = (
                    type(value) is bool,
                    value is flag,
                    function is root.backends.kleidiai.is_available,
                )
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))

        threads = [
            threading.Thread(target=worker, args=(index,))
            for index in range(worker_count)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        return (
            any(thread.is_alive() for thread in threads),
            errors,
            results,
        )

    def test_thread_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.threaded_contract(torch),
            self.threaded_contract(reference_torch),
        )

    def reload_contract(self, root):
        parent = root.backends
        module = parent.kleidiai
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
            self.fail("a stale KleidiAI availability query remained pickleable")

        return (
            reloaded is module,
            module.__dict__ is namespace,
            parent.kleidiai is module,
            sys.modules[module.__name__] is module,
            old_function is not new_function,
            new_function() is root._C._has_kleidiai,
            copy.copy(new_function) is new_function,
            copy.deepcopy(new_function) is new_function,
            pickle.loads(pickle.dumps(new_function)) is new_function,
            stale_pickle_error,
        )

    def test_reload_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_contract(torch),
            self.reload_contract(reference_torch),
        )
        actual = torch.backends.kleidiai.is_available
        expected = reference_torch.backends.kleidiai.is_available
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.backends.kleidiai.is_available
        expected = reference_torch.backends.kleidiai.is_available
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

    def test_supported_surface_does_not_claim_kernels_or_cpu_dispatch(self):
        actual = torch.backends.kleidiai
        expected = reference_torch.backends.kleidiai
        self.assertEqual(
            {name for name in vars(actual) if not name.startswith("_")},
            {name for name in vars(expected) if not name.startswith("_")},
        )
        self.assertEqual(
            {name for name in vars(actual) if not name.startswith("_")},
            {"is_available", "torch"},
        )
        self.assertEqual(torch.backends.cpu.get_cpu_capability(), "DEFAULT")


if __name__ == "__main__":
    unittest.main()
