import contextlib
import copy
import importlib
import inspect
import pickle
import threading
import types
import unittest

import torch_rs as torch
import torch_rs.compiler as compiler

try:
    import torch as reference_torch
    import torch.compiler as reference_compiler
except ImportError:
    reference_torch = None
    reference_compiler = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CompilerIsCompilingReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "compiler.is_compiling differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def state_outcome(self, module):
        function = module.compiler.is_compiling

        def query():
            before = module.is_grad_enabled()
            result = function()
            after = module.is_grad_enabled()
            return before, result, after

        main_states = [query()]
        with module.no_grad():
            main_states.append(query())
            with module.no_grad():
                main_states.append(query())
            main_states.append(query())
        main_states.append(query())

        worker_count = 8
        barrier = threading.Barrier(worker_count)
        worker_states = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = module.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    worker_states[index] = (query(), query())
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
        return main_states, worker_states

    def test_eager_grad_mode_and_thread_states_match_pytorch_2_13(self):
        self.assertEqual(
            self.state_outcome(torch),
            self.state_outcome(reference_torch),
        )

    def test_reference_eager_backend_bounds_the_unsupported_true_state(self):
        observed_states = []

        def function(value):
            expected_state = reference_compiler.is_compiling()
            actual_state = compiler.is_compiling()
            observed_states.append((expected_state, actual_state))
            if expected_state:
                value = value + 10
            else:
                value = value - 10
            if actual_state:
                value = value + 100
            return value

        self.assertIs(reference_compiler.is_compiling(), False)
        self.assertIs(compiler.is_compiling(), False)
        compiled = reference_torch.compile(function, backend="eager")
        first = compiled(reference_torch.tensor(2))
        second = compiled(reference_torch.tensor(3))

        self.assertEqual(first.item(), 12)
        self.assertEqual(second.item(), 13)
        self.assertEqual(observed_states, [(True, False), (True, False)])
        self.assertIs(reference_compiler.is_compiling(), False)
        self.assertIs(compiler.is_compiling(), False)

    def test_imports_exports_and_unsupported_surface_match_supported_scope(self):
        actual_imported = importlib.import_module("torch_rs.compiler")
        expected_imported = importlib.import_module("torch.compiler")
        from torch_rs import compiler as from_package
        from torch_rs.compiler import is_compiling

        self.assertIs(torch.compiler, compiler)
        self.assertIs(actual_imported, compiler)
        self.assertIs(from_package, compiler)
        self.assertIs(is_compiling, compiler.is_compiling)
        self.assertIs(reference_torch.compiler, reference_compiler)
        self.assertIs(expected_imported, reference_compiler)
        self.assertEqual(compiler.__doc__, reference_compiler.__doc__)
        self.assertEqual(
            compiler.__all__,
            [name for name in reference_compiler.__all__ if name == "is_compiling"],
        )

        actual_wildcard = {}
        expected_wildcard = {}
        exec("from torch_rs.compiler import *", actual_wildcard)
        exec("from torch.compiler import *", expected_wildcard)
        self.assertIs(actual_wildcard["is_compiling"], compiler.is_compiling)
        self.assertIs(
            expected_wildcard["is_compiling"], reference_compiler.is_compiling
        )
        self.assertEqual(
            {name for name in actual_wildcard if name != "__builtins__"},
            {"is_compiling"},
        )

        self.assertFalse(hasattr(torch, "compile"))
        self.assertTrue(hasattr(reference_torch, "compile"))
        self.assertFalse(hasattr(torch, "export"))
        self.assertTrue(hasattr(reference_torch, "export"))
        for name in reference_compiler.__all__:
            if name == "is_compiling":
                continue
            with self.subTest(name=name):
                self.assertFalse(hasattr(compiler, name))
                self.assertNotIn(name, actual_wildcard)
                self.assertIn(name, expected_wildcard)

        self.assertNotIn("compiler", torch.__all__)
        self.assertNotIn("compiler", reference_torch.__all__)
        self.assertFalse(hasattr(torch, "is_compiling"))
        self.assertFalse(hasattr(reference_torch, "is_compiling"))

    def test_signature_annotations_documentation_module_and_pickle_match(self):
        actual = compiler.is_compiling
        expected = reference_compiler.is_compiling

        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertIs(inspect.getmodule(actual), compiler)
        self.assertIs(inspect.getmodule(expected), reference_compiler)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )
        self.assertEqual(inspect.signature(actual), inspect.signature(expected))

        for operation in (copy.copy, copy.deepcopy):
            with self.subTest(operation=operation.__name__):
                self.assertIs(operation(actual), actual)
                self.assertIs(operation(expected), expected)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                actual_payload = pickle.dumps(actual, protocol=protocol)
                expected_payload = pickle.dumps(expected, protocol=protocol)
                self.assertIn(b"torch_rs.compiler", actual_payload)
                self.assertIn(b"torch.compiler", expected_payload)
                self.assertIs(pickle.loads(actual_payload), actual)
                self.assertIs(pickle.loads(expected_payload), expected)

    def test_argument_errors_match_pytorch_2_13(self):
        actual = compiler.is_compiling
        expected = reference_compiler.is_compiling
        cases = (
            (lambda: actual(None), lambda: expected(None)),
            (lambda: actual(None, None), lambda: expected(None, None)),
            (lambda: actual(enabled=True), lambda: expected(enabled=True)),
            (
                lambda: actual(None, enabled=True),
                lambda: expected(None, enabled=True),
            ),
            (lambda: actual(value=None), lambda: expected(value=None)),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
