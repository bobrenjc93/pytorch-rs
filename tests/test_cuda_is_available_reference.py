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
class CudaIsAvailableReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "cuda.is_available differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(
            type(actual_raised.exception), type(expected_raised.exception)
        )
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def threaded_outcome(self, function):
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                barrier.wait(timeout=10)
                results[index] = tuple(function() for _ in range(50))
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

    def test_canonical_imports_metadata_and_documentation_match_pytorch_2_13(self):
        actual_cuda = importlib.import_module("torch_rs.cuda")
        expected_cuda = importlib.import_module("torch.cuda")
        actual = actual_cuda.is_available
        expected = expected_cuda.is_available

        self.assertIs(torch.cuda, actual_cuda)
        self.assertIs(reference_torch.cuda, expected_cuda)
        self.assertEqual(
            actual_cuda.__name__.replace("torch_rs", "torch"),
            expected_cuda.__name__,
        )
        self.assertEqual(
            actual_cuda.__package__.replace("torch_rs", "torch"),
            expected_cuda.__package__,
        )
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )
        self.assertEqual(
            str(inspect.signature(actual)), str(inspect.signature(expected))
        )

        self.assertEqual(
            "cuda" in torch.__all__, "cuda" in reference_torch.__all__
        )
        self.assertEqual(actual_cuda.__all__, ["is_available"])
        self.assertEqual(expected_cuda.__all__.count("is_available"), 1)

    def test_cpu_backend_result_and_thread_stability_are_backend_specific(self):
        actual = torch.cuda.is_available
        expected = reference_torch.cuda.is_available
        expected_result = expected()

        self.assertIs(type(actual()), type(expected_result))
        self.assertIs(actual(), False)

        actual_results = self.threaded_outcome(actual)
        expected_results = self.threaded_outcome(expected)
        self.assertEqual(actual_results, [(False,) * 50] * 8)
        self.assertEqual(expected_results, [(expected_result,) * 50] * 8)

    def test_visible_cuda_hardware_does_not_change_cpu_backend_availability(self):
        if not reference_torch.cuda.is_available():
            self.skipTest("requires a visible CUDA accelerator")

        initialized_before = reference_torch.cuda.is_initialized()
        self.assertIs(torch.cuda.is_available(), False)
        self.assertEqual(reference_torch.cuda.is_initialized(), initialized_before)
        self.assertGreaterEqual(reference_torch.cuda.device_count(), 1)
        self.assertTrue(reference_torch.cuda.get_device_name(0))
        self.assertIs(reference_torch.cuda.is_available(), True)
        self.assertIs(torch.cuda.is_available(), False)

    def test_every_other_pytorch_cuda_export_remains_unsupported(self):
        unsupported = set(reference_torch.cuda.__all__) - {"is_available"}
        self.assertTrue(unsupported)
        for name in sorted(unsupported):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.cuda, name))

    def test_pickling_matches_canonical_module_ownership(self):
        actual = torch.cuda.is_available
        expected = reference_torch.cuda.is_available
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(actual, protocol=protocol)), actual
                )
                self.assertIs(
                    pickle.loads(pickle.dumps(expected, protocol=protocol)),
                    expected,
                )

    def test_argument_errors_match_pytorch_2_13(self):
        cases = (
            (
                lambda: torch.cuda.is_available(None),
                lambda: reference_torch.cuda.is_available(None),
            ),
            (
                lambda: torch.cuda.is_available(None, None),
                lambda: reference_torch.cuda.is_available(None, None),
            ),
            (
                lambda: torch.cuda.is_available(unexpected=True),
                lambda: reference_torch.cuda.is_available(unexpected=True),
            ),
            (
                lambda: torch.cuda.is_available(None, unexpected=True),
                lambda: reference_torch.cuda.is_available(None, unexpected=True),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
