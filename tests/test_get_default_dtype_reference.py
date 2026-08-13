import contextlib
import inspect
import threading
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class GetDefaultDTypeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "get_default_dtype differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def default_dtype_outcome(self, module):
        canonical = module.float32
        values = (
            module.get_default_dtype(),
            module.get_default_dtype(),
            module.float,
            module.tensor(1.25).dtype,
            module.zeros((2, 0, 3)).dtype,
            module.ones((2, 3)).dtype,
            module.eye(3).dtype,
            module.full((2,), 1.25).dtype,
        )
        return tuple(value is canonical for value in values), str(values[0])

    def threaded_outcome(self, module):
        canonical = module.float32
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = module.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    first = module.get_default_dtype()
                    results[index] = (
                        first is canonical,
                        module.tensor(1.25).dtype is canonical,
                        module.zeros((0,)).dtype is canonical,
                        module.get_default_dtype() is first,
                        module.is_grad_enabled(),
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

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        return results

    def test_canonical_factory_dtype_matches_pytorch_2_13(self):
        self.assertEqual(
            self.default_dtype_outcome(torch),
            self.default_dtype_outcome(reference_torch),
        )

    def test_grad_context_and_thread_stability_matches_pytorch_2_13(self):
        self.assertEqual(
            self.threaded_outcome(torch),
            self.threaded_outcome(reference_torch),
        )

        for module in (torch, reference_torch):
            canonical = module.float32
            with module.no_grad():
                self.assertIs(module.get_default_dtype(), canonical)
                with module.no_grad():
                    self.assertIs(module.get_default_dtype(), canonical)
            self.assertIs(module.get_default_dtype(), canonical)

    def test_callable_metadata_matches_pytorch_2_13(self):
        actual = torch.get_default_dtype
        expected = reference_torch.get_default_dtype
        self.assertIs(type(actual), types.BuiltinFunctionType)
        self.assertIs(type(expected), types.BuiltinFunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__text_signature__, expected.__text_signature__)
        self.assertEqual(actual.__module__, torch.tensor.__module__)
        self.assertEqual(expected.__module__, reference_torch.tensor.__module__)
        self.assertEqual(
            "get_default_dtype" in torch.__all__,
            "get_default_dtype" in reference_torch.__all__,
        )
        for function in (actual, expected):
            with self.assertRaises(ValueError):
                inspect.signature(function)

    def test_no_argument_errors_match_pytorch_2_13(self):
        cases = (
            (
                lambda: torch.get_default_dtype(None),
                lambda: reference_torch.get_default_dtype(None),
            ),
            (
                lambda: torch.get_default_dtype(None, None),
                lambda: reference_torch.get_default_dtype(None, None),
            ),
            (
                lambda: torch.get_default_dtype(dtype=None),
                lambda: reference_torch.get_default_dtype(dtype=None),
            ),
            (
                lambda: torch.get_default_dtype(None, dtype=None),
                lambda: reference_torch.get_default_dtype(None, dtype=None),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
