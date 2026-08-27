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

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class _CallableBackend:
    def __call__(self, graph_module, example_inputs):
        return graph_module.forward


class _StringBackend(str):
    pass


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CompilerSetDefaultBackendReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "compiler.set_default_backend differentials require pinned "
                "PyTorch 2.13.0"
            )

    def setUp(self):
        self.original_actual = torch.compiler.get_default_backend()
        self.original_expected = reference_torch.compiler.get_default_backend()
        torch.compiler.set_default_backend(None)
        reference_torch.compiler.set_default_backend(None)

    def tearDown(self):
        torch.compiler.set_default_backend(self.original_actual)
        reference_torch.compiler.set_default_backend(self.original_expected)

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

    def threaded_outcome(self, module):
        first_backend = _CallableBackend()
        second_backend = _CallableBackend()
        ready = threading.Event()
        updated = threading.Event()
        observations = []
        errors = []

        module.compiler.set_default_backend(first_backend)

        def observer():
            try:
                observations.append(
                    module.compiler.get_default_backend() is first_backend
                )
                ready.set()
                if not updated.wait(timeout=10):
                    raise RuntimeError("timed out waiting for backend update")
                observations.append(
                    module.compiler.get_default_backend() is second_backend
                )
                observations.append(module.compiler.reset() is None)
                observations.append(
                    module.compiler.get_default_backend() is second_backend
                )
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))

        thread = threading.Thread(target=observer)
        thread.start()
        self.assertTrue(ready.wait(timeout=10))
        set_result = module.compiler.set_default_backend(second_backend)
        updated.set()
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        return set_result is None, observations

    def test_accepted_values_identity_reset_and_threads_match_pytorch_2_13(self):
        def function_backend(graph_module, example_inputs):
            return graph_module.forward

        values = (
            "",
            "".join(("ea", "ger")),
            _StringBackend("custom"),
            function_backend,
            _CallableBackend(),
            _CallableBackend,
            len,
        )

        for backend in values:
            with self.subTest(backend=backend):
                actual_result = torch.compiler.set_default_backend(backend)
                expected_result = reference_torch.compiler.set_default_backend(backend)
                self.assertIs(actual_result, expected_result)
                self.assertIs(actual_result, None)
                self.assertIs(torch.compiler.get_default_backend(), backend)
                self.assertIs(
                    reference_torch.compiler.get_default_backend(),
                    backend,
                )

                self.assertIs(torch.compiler.reset(), None)
                self.assertIs(reference_torch.compiler.reset(), None)
                self.assertIs(torch.compiler.get_default_backend(), backend)
                self.assertIs(
                    reference_torch.compiler.get_default_backend(),
                    backend,
                )

        self.assertIs(torch.compiler.set_default_backend(None), None)
        self.assertIs(reference_torch.compiler.set_default_backend(None), None)
        self.assertEqual(torch.compiler.get_default_backend(), "inductor")
        self.assertEqual(
            reference_torch.compiler.get_default_backend(),
            "inductor",
        )

        self.assertEqual(
            self.threaded_outcome(torch),
            self.threaded_outcome(reference_torch),
        )

    def test_invalid_values_and_call_shape_errors_match_pytorch_2_13(self):
        actual = torch.compiler.set_default_backend
        expected = reference_torch.compiler.set_default_backend
        marker = _CallableBackend()
        actual(marker)
        expected(marker)

        for value in (False, 0, 1.5, [], {}, object(), _CallableBackend()):
            if callable(value):
                continue
            with self.subTest(value=value):
                self.assert_error_matches(
                    lambda value=value: actual(value),
                    lambda value=value: expected(value),
                )
                self.assertIs(torch.compiler.get_default_backend(), marker)
                self.assertIs(
                    reference_torch.compiler.get_default_backend(),
                    marker,
                )

        cases = (
            (lambda: actual(), lambda: expected()),
            (lambda: actual(None, None), lambda: expected(None, None)),
            (lambda: actual(value=None), lambda: expected(value=None)),
            (
                lambda: actual(None, backend=None),
                lambda: expected(None, backend=None),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)
                self.assertIs(torch.compiler.get_default_backend(), marker)
                self.assertIs(
                    reference_torch.compiler.get_default_backend(),
                    marker,
                )

    def test_signature_annotations_documentation_and_identity_match(self):
        actual_compiler = importlib.import_module("torch_rs.compiler")
        expected_compiler = importlib.import_module("torch.compiler")

        for name in ("set_default_backend", "reset"):
            with self.subTest(name=name):
                actual = getattr(actual_compiler, name)
                expected = getattr(expected_compiler, name)
                self.assertIs(type(actual), types.FunctionType)
                self.assertIs(type(expected), types.FunctionType)
                self.assertEqual(
                    str(inspect.signature(actual)),
                    str(inspect.signature(expected)),
                )
                self.assertEqual(actual.__annotations__, expected.__annotations__)
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
                self.assertIs(inspect.getmodule(actual), actual_compiler)
                self.assertIs(inspect.getmodule(expected), expected_compiler)
                self.assertEqual(actual.__doc__, expected.__doc__)
                self.assertEqual(actual.__defaults__, expected.__defaults__)
                self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
                self.assertEqual(actual.__dict__, expected.__dict__)
                self.assertEqual(
                    hasattr(actual, "__text_signature__"),
                    hasattr(expected, "__text_signature__"),
                )

    def test_exports_copy_and_pickle_match_pytorch_2_13(self):
        actual_compiler = torch.compiler
        expected_compiler = reference_torch.compiler
        supported = {
            "assume_constant_result",
            "reset",
            "disable",
            "set_default_backend",
            "get_default_backend",
            "cudagraph_mark_step_begin",
            "is_compiling",
            "is_dynamo_compiling",
            "is_exporting",
            "keep_portable_guards_unsafe",
            "skip_guard_on_inbuilt_nn_modules_unsafe",
            "skip_guard_on_all_nn_modules_unsafe",
            "skip_guard_on_globals_unsafe",
            "skip_all_guards_unsafe",
        }
        self.assertEqual(
            actual_compiler.__all__,
            [name for name in expected_compiler.__all__ if name in supported],
        )

        for module in (actual_compiler, expected_compiler):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            for name in supported:
                self.assertIs(namespace[name], getattr(module, name))

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            for name in ("compiler", "reset", "set_default_backend"):
                self.assertNotIn(name, namespace)

        for name in ("reset", "set_default_backend"):
            actual = getattr(actual_compiler, name)
            expected = getattr(expected_compiler, name)
            self.assertIs(copy.copy(actual), actual)
            self.assertIs(copy.copy(expected), expected)
            self.assertIs(copy.deepcopy(actual), actual)
            self.assertIs(copy.deepcopy(expected), expected)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(name=name, protocol=protocol):
                    self.assertIs(
                        pickle.loads(pickle.dumps(actual, protocol)),
                        actual,
                    )
                    self.assertIs(
                        pickle.loads(pickle.dumps(expected, protocol)),
                        expected,
                    )
                    self.assertEqual(
                        self.pickle_shape(actual, protocol),
                        self.pickle_shape(expected, protocol),
                    )

    def test_reload_and_reimport_semantics_match_pytorch_2_13(self):
        for package in (torch, reference_torch):
            with self.subTest(package=package.__name__):
                original_module = package.compiler
                old_getter = original_module.get_default_backend
                old_setter = original_module.set_default_backend
                first_backend = _CallableBackend()
                second_backend = _CallableBackend()
                old_setter(first_backend)

                try:
                    self.assertIs(importlib.reload(original_module), original_module)
                    self.assertIs(package.compiler, original_module)
                    self.assertIs(old_getter(), first_backend)
                    self.assertIs(
                        original_module.get_default_backend(),
                        first_backend,
                    )

                    module_name = original_module.__name__
                    self.assertIs(sys.modules.pop(module_name), original_module)
                    replacement_module = importlib.import_module(module_name)
                    self.assertIsNot(replacement_module, original_module)
                    self.assertIs(package.compiler, replacement_module)
                    self.assertIs(
                        replacement_module.get_default_backend(),
                        first_backend,
                    )
                    self.assertIs(
                        replacement_module.set_default_backend(second_backend),
                        None,
                    )
                    self.assertIs(old_getter(), second_backend)
                    self.assertIs(old_setter(None), None)
                    self.assertEqual(
                        replacement_module.get_default_backend(),
                        "inductor",
                    )
                finally:
                    sys.modules[original_module.__name__] = original_module
                    package.compiler = original_module

    def test_compilation_and_backend_registration_remain_unsupported(self):
        self.assertTrue(callable(reference_torch.compile))
        self.assertTrue(callable(reference_torch.compiler.compile))
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch.compiler, "compile"))
        self.assertFalse(hasattr(torch.compiler, "register_backend"))


if __name__ == "__main__":
    unittest.main()
