import contextlib
import copy
import importlib
import inspect
import pickle
import threading
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


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

    def supported_state_outcome(self, module):
        function = module.compiler.is_compiling

        def query_outcome():
            before = module.is_grad_enabled()
            result = function()
            after = module.is_grad_enabled()
            return before, result is False, after

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

    def test_supported_eager_no_grad_and_thread_states_match_pytorch_2_13(self):
        self.assertEqual(
            self.supported_state_outcome(torch),
            self.supported_state_outcome(reference_torch),
        )

    def test_reference_eager_backend_compile_bounds_the_unsupported_true_state(self):
        actual_function = torch.compiler.is_compiling
        expected_function = reference_torch.compiler.is_compiling

        def state_branch(value):
            if expected_function():
                return value + 1
            return value - 1

        value = reference_torch.tensor(1)
        self.assertIs(actual_function(), False)
        self.assertIs(expected_function(), False)
        self.assertEqual(state_branch(value).item(), 0)

        compiled = reference_torch.compile(
            state_branch,
            backend="eager",
            fullgraph=True,
        )
        self.assertEqual(compiled(value).item(), 2)

        self.assertIs(actual_function(), False)
        self.assertIs(expected_function(), False)

    def test_function_contract_matches_pytorch_2_13(self):
        actual_compiler = importlib.import_module("torch_rs.compiler")
        expected_compiler = importlib.import_module("torch.compiler")
        actual = actual_compiler.is_compiling
        expected = expected_compiler.is_compiling

        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(inspect.signature(actual), inspect.signature(expected))
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )
        self.assertEqual(
            hasattr(actual, "__wrapped__"), hasattr(expected, "__wrapped__")
        )
        self.assertEqual(
            hasattr(actual, "__signature__"), hasattr(expected, "__signature__")
        )
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertIs(inspect.getmodule(actual), actual_compiler)
        self.assertIs(inspect.getmodule(expected), expected_compiler)

        for operation in (copy.copy, copy.deepcopy):
            with self.subTest(operation=operation.__name__):
                self.assertIs(operation(actual), actual)
                self.assertIs(operation(expected), expected)

        self.assertEqual(
            actual_compiler.__all__,
            [name for name in expected_compiler.__all__ if name == "is_compiling"],
        )
        self.assertEqual(
            torch.__all__.count("compiler"),
            reference_torch.__all__.count("compiler"),
        )

        for module, function in (
            (actual_compiler, actual),
            (expected_compiler, expected),
        ):
            wildcard_namespace = {}
            exec(f"from {module.__name__} import *", wildcard_namespace)
            self.assertIs(wildcard_namespace["is_compiling"], function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(module=module.__name__, protocol=protocol):
                    payload = pickle.dumps(function, protocol=protocol)
                    self.assertIn(module.__name__.encode(), payload)
                    self.assertIs(pickle.loads(payload), function)

        for module in (torch, reference_torch):
            wildcard_namespace = {}
            exec(f"from {module.__name__} import *", wildcard_namespace)
            self.assertNotIn("compiler", wildcard_namespace)

    def test_call_errors_match_pytorch_2_13(self):
        cases = (
            (
                lambda: torch.compiler.is_compiling(None),
                lambda: reference_torch.compiler.is_compiling(None),
            ),
            (
                lambda: torch.compiler.is_compiling(None, None),
                lambda: reference_torch.compiler.is_compiling(None, None),
            ),
            (
                lambda: torch.compiler.is_compiling(unexpected=True),
                lambda: reference_torch.compiler.is_compiling(unexpected=True),
            ),
            (
                lambda: torch.compiler.is_compiling(None, unexpected=True),
                lambda: reference_torch.compiler.is_compiling(
                    None, unexpected=True
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_other_compiler_apis_remain_outside_the_supported_surface(self):
        actual_compiler = torch.compiler
        expected_compiler = reference_torch.compiler

        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch, "export"))
        for name in expected_compiler.__all__:
            if name == "is_compiling":
                continue
            with self.subTest(name=name):
                self.assertFalse(hasattr(actual_compiler, name))


if __name__ == "__main__":
    unittest.main()
