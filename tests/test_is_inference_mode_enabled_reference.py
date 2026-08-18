import contextlib
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
class IsInferenceModeEnabledReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "is_inference_mode_enabled differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def supported_state_outcome(self, module):
        function = module.is_inference_mode_enabled

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

    def test_supported_default_and_no_grad_states_match_pytorch_2_13(self):
        self.assertEqual(
            self.supported_state_outcome(torch),
            self.supported_state_outcome(reference_torch),
        )

    def test_pytorch_inference_context_bounds_the_unsupported_true_state(self):
        actual_function = torch.is_inference_mode_enabled
        expected_function = reference_torch.is_inference_mode_enabled

        actual_states = [actual_function()]
        expected_states = [expected_function()]
        with reference_torch.inference_mode():
            actual_states.append(actual_function())
            expected_states.append(expected_function())
            self.assertIs(torch.is_grad_enabled(), True)
            self.assertIs(reference_torch.is_grad_enabled(), False)
        actual_states.append(actual_function())
        expected_states.append(expected_function())

        for state in actual_states:
            self.assertIs(state, False)
        self.assertIs(expected_states[0], False)
        self.assertIs(expected_states[1], True)
        self.assertIs(expected_states[2], False)
        self.assertIs(torch.is_grad_enabled(), True)
        self.assertIs(reference_torch.is_grad_enabled(), True)

    def test_builtin_contract_matches_pytorch_2_13(self):
        actual = torch.is_inference_mode_enabled
        expected = reference_torch.is_inference_mode_enabled

        self.assertIs(type(actual), types.BuiltinFunctionType)
        self.assertIs(type(expected), types.BuiltinFunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs.torch_rs", "torch"),
            expected.__module__,
        )
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__text_signature__, expected.__text_signature__)
        self.assertEqual(repr(actual), repr(expected))
        self.assertIs(actual.__self__, torch._C)
        self.assertIs(expected.__self__, reference_torch._C)
        self.assertIs(torch._C.is_inference_mode_enabled, actual)
        self.assertIs(reference_torch._C.is_inference_mode_enabled, expected)
        for function in (actual, expected):
            assert_no_argument_signature(self, function, "()")

        self.assertEqual(
            torch.__all__.count("is_inference_mode_enabled"),
            reference_torch.__all__.count("is_inference_mode_enabled"),
        )
        for module, function in ((torch, actual), (reference_torch, expected)):
            wildcard_namespace = {}
            exec(f"from {module.__name__} import *", wildcard_namespace)
            self.assertIs(wildcard_namespace["is_inference_mode_enabled"], function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(module=module.__name__, protocol=protocol):
                    restored = pickle.loads(pickle.dumps(function, protocol=protocol))
                    self.assertIs(restored, function)

    def test_no_argument_errors_match_pytorch_2_13(self):
        cases = (
            (
                lambda: torch.is_inference_mode_enabled(None),
                lambda: reference_torch.is_inference_mode_enabled(None),
            ),
            (
                lambda: torch.is_inference_mode_enabled(None, None),
                lambda: reference_torch.is_inference_mode_enabled(None, None),
            ),
            (
                lambda: torch.is_inference_mode_enabled(enabled=True),
                lambda: reference_torch.is_inference_mode_enabled(enabled=True),
            ),
            (
                lambda: torch.is_inference_mode_enabled(None, enabled=True),
                lambda: reference_torch.is_inference_mode_enabled(
                    None, enabled=True
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
