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


FUNCTION_DOC = """
is_inference_mode_enabled() -> (bool)

Returns True if inference mode is currently enabled.
"""


class IsInferenceModeEnabledTests(unittest.TestCase):
    def test_default_false_is_exact_and_does_not_change_grad_mode(self):
        function = torch.is_inference_mode_enabled

        def assert_query_preserves_grad_mode(expected_grad_state):
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)
            self.assertIs(function(), False)
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)

        assert_query_preserves_grad_mode(True)
        with torch.no_grad():
            assert_query_preserves_grad_mode(False)
            with torch.no_grad():
                assert_query_preserves_grad_mode(False)
            assert_query_preserves_grad_mode(False)
        assert_query_preserves_grad_mode(True)

    def test_default_false_is_stable_across_threads_and_no_grad(self):
        function = torch.is_inference_mode_enabled
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = torch.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    results[index] = (
                        torch.is_grad_enabled(),
                        function(),
                        torch.is_grad_enabled(),
                        function(),
                        torch.is_grad_enabled(),
                    )
            except BaseException as error:
                errors.append(error)

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
        for index, result in enumerate(results):
            expected_grad_state = index % 2 == 0
            self.assertIs(result[0], expected_grad_state)
            self.assertIs(result[1], False)
            self.assertIs(result[2], expected_grad_state)
            self.assertIs(result[3], False)
            self.assertIs(result[4], expected_grad_state)

    def test_builtin_ownership_documentation_exports_and_pickling(self):
        function = torch.is_inference_mode_enabled
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "is_inference_mode_enabled")
        self.assertEqual(function.__qualname__, "is_inference_mode_enabled")
        self.assertEqual(function.__module__, torch.tensor.__module__)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertEqual(repr(function), "<built-in function is_inference_mode_enabled>")
        self.assertIs(function.__self__, torch._C)
        self.assertIs(torch._C.is_inference_mode_enabled, function)
        assert_no_argument_signature(self, function, "()")

        self.assertEqual(torch.__all__.count("is_inference_mode_enabled"), 1)
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["is_inference_mode_enabled"], function)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                restored = pickle.loads(pickle.dumps(function, protocol=protocol))
                self.assertIs(restored, function)

    def test_rejects_all_arguments_with_pytorch_2_13_errors(self):
        function = torch.is_inference_mode_enabled
        cases = (
            (
                lambda: function(None),
                "torch.is_inference_mode_enabled() takes no arguments (1 given)",
            ),
            (
                lambda: function(None, None),
                "torch.is_inference_mode_enabled() takes no arguments (2 given)",
            ),
            (
                lambda: function(enabled=True),
                "torch.is_inference_mode_enabled() takes no keyword arguments",
            ),
            (
                lambda: function(None, enabled=True),
                "torch.is_inference_mode_enabled() takes no keyword arguments",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)


if __name__ == "__main__":
    unittest.main()
