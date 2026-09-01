import copy
import importlib
import inspect
import pickle
import pickletools
import sys
import threading
import types
import typing
import unittest
from unittest import mock

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class MhaFastpathReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "backends.mha differentials require pinned PyTorch 2.13.0"
            )

    def setUp(self):
        self.actual = importlib.reload(
            importlib.import_module("torch_rs.backends.mha")
        )
        self.expected = importlib.reload(
            importlib.import_module("torch.backends.mha")
        )

    def tearDown(self):
        importlib.reload(self.actual)
        importlib.reload(self.expected)

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

    def state_contract(self, root, module):
        outcomes = [
            module.get_fastpath_enabled() is True,
            module._is_fastpath_enabled is True,
        ]
        for value in (False, None, 0, [], object()):
            outcomes.append(module.set_fastpath_enabled(value) is None)
            outcomes.append(module._is_fastpath_enabled is value)
            outcomes.append(module.get_fastpath_enabled() is value)

        stored = object()
        module.set_fastpath_enabled(stored)
        with mock.patch.object(root.jit, "is_scripting", return_value=True) as probe:
            outcomes.append(module.get_fastpath_enabled() is True)
            outcomes.append(probe.call_args_list)
        outcomes.append(module._is_fastpath_enabled is stored)
        outcomes.append(module.get_fastpath_enabled() is stored)
        return outcomes

    def thread_contract(self, module):
        initial = object()
        worker_value = object()
        main_value = object()
        worker_written = threading.Event()
        main_written = threading.Event()
        outcomes = {}
        errors = []

        module.set_fastpath_enabled(initial)

        def worker():
            try:
                outcomes["initial"] = module.get_fastpath_enabled() is initial
                outcomes["setter"] = (
                    module.set_fastpath_enabled(worker_value) is None
                )
                worker_written.set()
                if not main_written.wait(timeout=10):
                    raise TimeoutError("main thread did not publish its state")
                outcomes["final"] = module.get_fastpath_enabled() is main_value
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))
                worker_written.set()

        thread = threading.Thread(target=worker)
        thread.start()
        worker_ready = worker_written.wait(timeout=10)
        main_saw_worker = module.get_fastpath_enabled() is worker_value
        main_setter = module.set_fastpath_enabled(main_value) is None
        main_written.set()
        thread.join(timeout=10)
        return (
            worker_ready,
            main_saw_worker,
            main_setter,
            not thread.is_alive(),
            errors,
            outcomes,
        )

    def reload_contract(self, root, module):
        old_getter = module.get_fastpath_enabled
        old_setter = module.set_fastpath_enabled
        namespace = module.__dict__
        module.set_fastpath_enabled(object())
        reloaded = importlib.reload(module)
        return (
            reloaded is module,
            module.__dict__ is namespace,
            root.backends.mha is module,
            sys.modules[module.__name__] is module,
            module._is_fastpath_enabled is True,
            module.get_fastpath_enabled() is True,
            module.get_fastpath_enabled is not old_getter,
            module.set_fastpath_enabled is not old_setter,
        )

    def test_default_identity_setter_scripting_and_threads_match_pytorch_2_13(self):
        self.assertEqual(
            self.state_contract(torch, self.actual),
            self.state_contract(reference_torch, self.expected),
        )
        self.assertEqual(
            self.thread_contract(self.actual),
            self.thread_contract(self.expected),
        )

    def test_signature_documentation_and_module_identity_match_pytorch_2_13(self):
        actual = self.actual
        expected = self.expected

        self.assertIs(torch.backends.mha, actual)
        self.assertIs(reference_torch.backends.mha, expected)
        self.assertIs(sys.modules[actual.__name__], actual)
        self.assertIs(sys.modules[expected.__name__], expected)
        self.assertIs(type(actual), types.ModuleType)
        self.assertIs(type(expected), types.ModuleType)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(hasattr(actual, "__all__"), hasattr(expected, "__all__"))
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(
            {name for name in vars(actual) if not name.startswith("_")},
            {name for name in vars(expected) if not name.startswith("_")},
        )
        self.assertIs(actual.torch, torch)
        self.assertIs(expected.torch, reference_torch)

        for name in ("get_fastpath_enabled", "set_fastpath_enabled"):
            with self.subTest(function=name):
                actual_function = getattr(actual, name)
                expected_function = getattr(expected, name)
                self.assertIs(type(actual_function), types.FunctionType)
                self.assertIs(type(expected_function), types.FunctionType)
                self.assertEqual(
                    str(inspect.signature(actual_function)),
                    str(inspect.signature(expected_function)),
                )
                self.assertEqual(
                    actual_function.__annotations__,
                    expected_function.__annotations__,
                )
                self.assertEqual(
                    typing.get_type_hints(actual_function),
                    typing.get_type_hints(expected_function),
                )
                self.assertEqual(actual_function.__name__, expected_function.__name__)
                self.assertEqual(
                    actual_function.__qualname__, expected_function.__qualname__
                )
                self.assertEqual(
                    actual_function.__module__.replace("torch_rs", "torch"),
                    expected_function.__module__,
                )
                self.assertIs(inspect.getmodule(actual_function), actual)
                self.assertIs(inspect.getmodule(expected_function), expected)
                self.assertEqual(actual_function.__doc__, expected_function.__doc__)
                self.assertEqual(
                    actual_function.__defaults__, expected_function.__defaults__
                )
                self.assertEqual(
                    actual_function.__kwdefaults__, expected_function.__kwdefaults__
                )
                self.assertEqual(actual_function.__dict__, expected_function.__dict__)
                self.assertEqual(
                    hasattr(actual_function, "__text_signature__"),
                    hasattr(expected_function, "__text_signature__"),
                )
                self.assertEqual(
                    actual_function.__code__.co_names,
                    expected_function.__code__.co_names,
                )
                self.assertEqual(
                    actual_function.__code__.co_freevars,
                    expected_function.__code__.co_freevars,
                )
                self.assertEqual(
                    actual_function.__code__.co_cellvars,
                    expected_function.__code__.co_cellvars,
                )

    def test_imports_wildcards_copying_and_pickling_match_pytorch_2_13(self):
        actual_backends = importlib.import_module("torch_rs.backends")
        expected_backends = importlib.import_module("torch.backends")
        actual = self.actual
        expected = self.expected
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

        for package_name, module in (("torch_rs", actual), ("torch", expected)):
            backend_import = {}
            exec(f"from {package_name}.backends import mha", backend_import)
            self.assertIs(backend_import["mha"], module)
            for name in ("get_fastpath_enabled", "set_fastpath_enabled"):
                function_import = {}
                exec(
                    f"from {package_name}.backends.mha import {name}",
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
                if name in supported_backends
            },
        )

        actual_child_wildcard = {}
        expected_child_wildcard = {}
        exec("from torch_rs.backends.mha import *", actual_child_wildcard)
        exec("from torch.backends.mha import *", expected_child_wildcard)
        self.assertEqual(
            {name for name in actual_child_wildcard if not name.startswith("__")},
            {name for name in expected_child_wildcard if not name.startswith("__")},
        )
        self.assertIs(actual_child_wildcard["torch"], torch)
        self.assertIs(expected_child_wildcard["torch"], reference_torch)

        for root in (torch, reference_torch):
            namespace = {}
            exec(f"from {root.__name__} import *", namespace)
            self.assertNotIn("backends", namespace)
            self.assertNotIn("mha", namespace)
            self.assertFalse(hasattr(root, "mha"))

        for name in ("get_fastpath_enabled", "set_fastpath_enabled"):
            actual_function = getattr(actual, name)
            expected_function = getattr(expected, name)
            self.assertIs(copy.copy(actual_function), actual_function)
            self.assertIs(copy.copy(expected_function), expected_function)
            self.assertIs(copy.deepcopy(actual_function), actual_function)
            self.assertIs(copy.deepcopy(expected_function), expected_function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(function=name, protocol=protocol):
                    self.assertIs(
                        pickle.loads(pickle.dumps(actual_function, protocol)),
                        actual_function,
                    )
                    self.assertIs(
                        pickle.loads(pickle.dumps(expected_function, protocol)),
                        expected_function,
                    )
                    self.assertEqual(
                        self.pickle_shape(actual_function, protocol),
                        self.pickle_shape(expected_function, protocol),
                    )

        self.assertIs(actual_backends.mha, actual)
        self.assertIs(expected_backends.mha, expected)

    def test_reload_reset_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_contract(torch, self.actual),
            self.reload_contract(reference_torch, self.expected),
        )

    def test_argument_forms_and_errors_match_pytorch_2_13(self):
        actual = self.actual
        expected = self.expected

        actual_value = object()
        expected_value = object()
        self.assertIsNone(actual.set_fastpath_enabled(value=actual_value))
        self.assertIsNone(expected.set_fastpath_enabled(value=expected_value))
        self.assertIs(actual.get_fastpath_enabled(), actual_value)
        self.assertIs(expected.get_fastpath_enabled(), expected_value)

        cases = (
            (
                lambda: actual.get_fastpath_enabled(None),
                lambda: expected.get_fastpath_enabled(None),
            ),
            (
                lambda: actual.get_fastpath_enabled(enabled=True),
                lambda: expected.get_fastpath_enabled(enabled=True),
            ),
            (
                lambda: actual.set_fastpath_enabled(),
                lambda: expected.set_fastpath_enabled(),
            ),
            (
                lambda: actual.set_fastpath_enabled(True, False),
                lambda: expected.set_fastpath_enabled(True, False),
            ),
            (
                lambda: actual.set_fastpath_enabled(enabled=True),
                lambda: expected.set_fastpath_enabled(enabled=True),
            ),
            (
                lambda: actual.set_fastpath_enabled(True, value=False),
                lambda: expected.set_fastpath_enabled(True, value=False),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_attention_execution_surface_remains_explicitly_unsupported(self):
        for name in (
            "MultiheadAttention",
            "Transformer",
            "TransformerDecoder",
            "TransformerEncoder",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.nn, name))
                self.assertTrue(hasattr(reference_torch.nn, name))

        self.assertFalse(
            hasattr(torch.nn.functional, "multi_head_attention_forward")
        )
        self.assertTrue(
            hasattr(reference_torch.nn.functional, "multi_head_attention_forward")
        )
        self.assertFalse(hasattr(torch, "_native_multi_head_attention"))
        self.assertTrue(hasattr(reference_torch, "_native_multi_head_attention"))


if __name__ == "__main__":
    unittest.main()
