import contextlib
import copy
import importlib
import inspect
import pickle
import pickletools
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
class CompilerGetDefaultBackendReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "compiler.get_default_backend differentials require pinned "
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

    def supported_state_outcome(self, module):
        function = module.compiler.get_default_backend

        def query_outcome():
            before = module.is_grad_enabled()
            first = function()
            middle = module.is_grad_enabled()
            second = function()
            after = module.is_grad_enabled()
            return (
                before,
                type(first) is str,
                first,
                middle,
                type(second) is str,
                second,
                after,
            )

        states = [query_outcome()]
        with module.no_grad():
            states.append(query_outcome())
            with module.no_grad():
                states.append(query_outcome())
            states.append(query_outcome())
        states.append(query_outcome())

        worker_count = 8
        barrier = threading.Barrier(worker_count)
        worker_states = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = module.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    worker_states[index] = query_outcome()
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

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        return states, worker_states

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

    def test_supported_default_threaded_and_grad_states_match_pytorch_2_13(self):
        actual_getter = torch.compiler.get_default_backend
        actual_setter = torch.compiler.set_default_backend
        expected_getter = reference_torch.compiler.get_default_backend
        expected_setter = reference_torch.compiler.set_default_backend
        original_actual = actual_getter()
        original_expected = expected_getter()

        try:
            actual_setter(None)
            expected_setter(None)
            self.assertEqual(
                self.supported_state_outcome(torch),
                self.supported_state_outcome(reference_torch),
            )
        finally:
            actual_setter(original_actual)
            expected_setter(original_expected)

        self.assertIs(actual_getter(), original_actual)
        self.assertIs(expected_getter(), original_expected)

    def test_setter_states_and_reset_match_pytorch_2_13(self):
        actual_getter = torch.compiler.get_default_backend
        actual_setter = torch.compiler.set_default_backend
        expected_getter = reference_torch.compiler.get_default_backend
        expected_setter = reference_torch.compiler.set_default_backend
        original_actual = actual_getter()
        original_expected = expected_getter()
        alternate_name = "".join(("ea", "ger"))

        def alternate_callable(graph_module, example_inputs):
            return graph_module.forward

        try:
            for value in (None, alternate_name, alternate_callable):
                with self.subTest(value=value):
                    self.assertIs(actual_setter(value), None)
                    self.assertIs(expected_setter(value), None)
                    if value is None:
                        self.assertEqual(actual_getter(), "inductor")
                        self.assertEqual(expected_getter(), "inductor")
                    else:
                        self.assertIs(actual_getter(), value)
                        self.assertIs(expected_getter(), value)

            self.assertIs(torch.compiler.reset(), None)
            self.assertIs(reference_torch.compiler.reset(), None)
            self.assertIs(actual_getter(), alternate_callable)
            self.assertIs(expected_getter(), alternate_callable)

            self.assertIs(actual_setter(None), None)
            self.assertIs(expected_setter(None), None)
            self.assertEqual(actual_getter(), "inductor")
            self.assertEqual(expected_getter(), "inductor")
        finally:
            actual_setter(original_actual)
            expected_setter(original_expected)

        self.assertIs(actual_getter(), original_actual)
        self.assertIs(expected_getter(), original_expected)

    def test_signature_annotations_documentation_and_identity_match(self):
        actual_compiler = importlib.import_module("torch_rs.compiler")
        expected_compiler = importlib.import_module("torch.compiler")
        actual = actual_compiler.get_default_backend
        expected = expected_compiler.get_default_backend

        self.assertIs(torch.compiler, actual_compiler)
        self.assertIs(reference_torch.compiler, expected_compiler)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual)),
            str(inspect.signature(expected)),
        )
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(typing.get_type_hints(actual), typing.get_type_hints(expected))
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
        actual = actual_compiler.get_default_backend
        expected = expected_compiler.get_default_backend
        supported = {
            "assume_constant_result",
            "reset",
            "disable",
            "set_default_backend",
            "get_default_backend",
            "set_enable_guard_collectives",
            "cudagraph_mark_step_begin",
            "is_compiling",
            "is_dynamo_compiling",
            "is_exporting",
            "keep_portable_guards_unsafe",
            "skip_guard_on_inbuilt_nn_modules_unsafe",
            "skip_guard_on_all_nn_modules_unsafe",
            "keep_tensor_guards_unsafe",
            "skip_guard_on_globals_unsafe",
            "skip_all_guards_unsafe",
        }

        self.assertEqual(
            actual_compiler.__all__,
            [name for name in expected_compiler.__all__ if name in supported],
        )
        self.assertEqual(
            torch.__all__.count("compiler"),
            reference_torch.__all__.count("compiler"),
        )
        self.assertEqual(
            torch.__all__.count("get_default_backend"),
            reference_torch.__all__.count("get_default_backend"),
        )

        for module in (actual_compiler, expected_compiler):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            for name in supported:
                self.assertIs(namespace[name], getattr(module, name))

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("compiler", namespace)
            self.assertNotIn("get_default_backend", namespace)

        for function in (actual, expected):
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(pickle.loads(pickle.dumps(expected, protocol)), expected)
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.compiler.get_default_backend
        expected = reference_torch.compiler.get_default_backend
        cases = (
            (lambda: actual(None), lambda: expected(None)),
            (lambda: actual(None, None), lambda: expected(None, None)),
            (
                lambda: actual(backend=None),
                lambda: expected(backend=None),
            ),
            (
                lambda: actual(None, backend=None),
                lambda: expected(None, backend=None),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_compilation_remains_deliberately_unsupported(self):
        self.assertTrue(callable(torch.compiler.reset))
        self.assertTrue(callable(torch.compiler.set_default_backend))
        self.assertTrue(callable(reference_torch.compile))
        self.assertTrue(callable(reference_torch.compiler.compile))
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch.compiler, "compile"))


if __name__ == "__main__":
    unittest.main()
