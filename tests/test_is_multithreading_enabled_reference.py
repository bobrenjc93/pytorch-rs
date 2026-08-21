import contextlib
import copy
import pickle
import threading
import types
import unittest

import torch_rs as torch

if __package__:
    from .signature_utils import assert_no_argument_signature
else:
    from signature_utils import assert_no_argument_signature

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class IsMultithreadingEnabledReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "is_multithreading_enabled differentials require pinned PyTorch 2.13.0"
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
        function = module.autograd.is_multithreading_enabled

        def query_outcome():
            before = module.is_grad_enabled()
            first = function()
            middle = module.is_grad_enabled()
            second = function()
            after = module.is_grad_enabled()
            return (
                before,
                first is True,
                middle,
                second is True,
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

    def test_supported_true_and_grad_states_match_pytorch_2_13(self):
        with reference_torch.autograd.set_multithreading_enabled(True):
            self.assertEqual(
                self.supported_state_outcome(torch),
                self.supported_state_outcome(reference_torch),
            )

    def test_reference_only_setter_context_bounds_unsupported_false_state(self):
        actual_function = torch.autograd.is_multithreading_enabled
        expected_function = reference_torch.autograd.is_multithreading_enabled
        previous_reference_state = expected_function()

        self.assertFalse(hasattr(torch.autograd, "set_multithreading_enabled"))
        with reference_torch.autograd.set_multithreading_enabled(True):
            self.assertIs(actual_function(), True)
            self.assertIs(expected_function(), True)

            with reference_torch.autograd.set_multithreading_enabled(False):
                self.assertIs(actual_function(), True)
                self.assertIs(expected_function(), False)
                with torch.no_grad(), reference_torch.no_grad():
                    self.assertIs(actual_function(), True)
                    self.assertIs(expected_function(), False)
                    self.assertIs(torch.is_grad_enabled(), False)
                    self.assertIs(reference_torch.is_grad_enabled(), False)

            self.assertIs(actual_function(), True)
            self.assertIs(expected_function(), True)

        self.assertIs(actual_function(), True)
        self.assertIs(expected_function(), previous_reference_state)

    def test_builtin_contract_matches_pytorch_2_13(self):
        actual = torch.autograd.is_multithreading_enabled
        expected = reference_torch.autograd.is_multithreading_enabled

        self.assertIs(type(actual), types.BuiltinFunctionType)
        self.assertIs(type(expected), types.BuiltinFunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs.torch_rs", "torch._C"),
            expected.__module__,
        )
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__text_signature__, expected.__text_signature__)
        self.assertEqual(
            hasattr(actual, "__annotations__"),
            hasattr(expected, "__annotations__"),
        )
        self.assertFalse(hasattr(actual, "__annotations__"))
        self.assertEqual(repr(actual), repr(expected))
        self.assertIs(actual.__self__, torch._C)
        self.assertIs(expected.__self__, reference_torch._C)
        self.assertIs(torch._C._is_multithreading_enabled, actual)
        self.assertIs(reference_torch._C._is_multithreading_enabled, expected)
        for function in (actual, expected):
            assert_no_argument_signature(self, function, "()")
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(function=function, protocol=protocol):
                    restored = pickle.loads(pickle.dumps(function, protocol=protocol))
                    self.assertIs(restored, function)

    def test_import_and_non_wildcard_behavior_matches_pytorch_2_13(self):
        actual = torch.autograd.is_multithreading_enabled
        expected = reference_torch.autograd.is_multithreading_enabled

        for module, function in (
            (torch.autograd, actual),
            (reference_torch.autograd, expected),
        ):
            self.assertNotIn("is_multithreading_enabled", module.__all__)
            wildcard_namespace = {}
            exec(f"from {module.__name__} import *", wildcard_namespace)
            self.assertNotIn("is_multithreading_enabled", wildcard_namespace)

            explicit_namespace = {}
            exec(
                f"from {module.__name__} import is_multithreading_enabled",
                explicit_namespace,
            )
            self.assertIs(explicit_namespace["is_multithreading_enabled"], function)

        for module in (torch, reference_torch):
            self.assertFalse(hasattr(module, "is_multithreading_enabled"))
            self.assertFalse(hasattr(module, "_is_multithreading_enabled"))
            self.assertNotIn("is_multithreading_enabled", module.__all__)
            self.assertNotIn("_is_multithreading_enabled", module.__all__)

        self.assertNotIn("_is_multithreading_enabled", torch._C.__all__)

    def test_no_argument_errors_match_pytorch_2_13(self):
        actual = torch.autograd.is_multithreading_enabled
        expected = reference_torch.autograd.is_multithreading_enabled
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
                self.assert_error_matches(actual_call, expected_call)

        self.assertIs(actual(**{}), True)
        self.assertIs(expected(**{}), True)

    def test_mutation_surfaces_remain_deliberately_absent(self):
        self.assertFalse(hasattr(torch._C, "_set_multithreading_enabled"))
        self.assertTrue(hasattr(reference_torch._C, "_set_multithreading_enabled"))
        self.assertFalse(hasattr(torch.autograd, "set_multithreading_enabled"))
        self.assertTrue(
            hasattr(reference_torch.autograd, "set_multithreading_enabled")
        )
        self.assertFalse(
            hasattr(torch.autograd.grad_mode, "set_multithreading_enabled")
        )
        self.assertTrue(
            hasattr(
                reference_torch.autograd.grad_mode,
                "set_multithreading_enabled",
            )
        )


if __name__ == "__main__":
    unittest.main()
