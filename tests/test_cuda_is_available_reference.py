import importlib
import inspect
import pickle
import sys
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
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def threaded_outcome(self, function):
        worker_count = 16
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                barrier.wait(timeout=10)
                results[index] = tuple(function() for _ in range(100))
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

    def test_canonical_namespace_and_supported_export_match_pytorch_2_13(self):
        actual = importlib.import_module("torch_rs.cuda")
        expected = importlib.import_module("torch.cuda")

        self.assertIs(torch.cuda, actual)
        self.assertIs(reference_torch.cuda, expected)
        self.assertIs(sys.modules["torch_rs.cuda"], actual)
        self.assertIs(sys.modules["torch.cuda"], expected)
        self.assertIs(type(actual), types.ModuleType)
        self.assertIs(type(expected), types.ModuleType)
        self.assertEqual(actual.__name__.replace("torch_rs", "torch"), expected.__name__)
        self.assertEqual(
            actual.__package__.replace("torch_rs", "torch"),
            expected.__package__,
        )
        self.assertEqual(
            actual.__spec__.name.replace("torch_rs", "torch"),
            expected.__spec__.name,
        )
        self.assertIn("is_available", actual.__all__)
        self.assertIn("is_available", expected.__all__)
        self.assertEqual(actual.__all__, ["is_available"])
        self.assertEqual(
            "cuda" in torch.__all__,
            "cuda" in reference_torch.__all__,
        )

        actual_wildcard = {}
        exec("from torch_rs.cuda import *", actual_wildcard)
        self.assertIs(actual_wildcard["is_available"], actual.is_available)
        self.assertEqual(
            {name for name in actual_wildcard if not name.startswith("_")},
            {"is_available"},
        )

        for unsupported in (
            "current_device",
            "device_count",
            "Event",
            "get_device_name",
            "init",
            "is_initialized",
            "set_device",
            "Stream",
            "synchronize",
        ):
            with self.subTest(unsupported=unsupported):
                self.assertFalse(hasattr(actual, unsupported))
                self.assertTrue(hasattr(expected, unsupported))

    def test_cpu_backend_result_and_thread_stability_have_reference_types(self):
        actual_function = torch.cuda.is_available
        expected_function = reference_torch.cuda.is_available
        actual_value = actual_function()
        expected_value = expected_function()

        self.assertIs(actual_value, False)
        self.assertIs(type(actual_value), bool)
        self.assertIs(type(expected_value), bool)

        for function, expected in (
            (actual_function, actual_value),
            (expected_function, expected_value),
        ):
            results = self.threaded_outcome(function)
            for result in results:
                self.assertEqual(result, (expected,) * 100)
                self.assertTrue(all(type(value) is bool for value in result))

    def test_visible_reference_cuda_does_not_change_the_cpu_backend_answer(self):
        initialized_before = reference_torch.cuda.is_initialized()
        self.assertIs(torch.cuda.is_available(), False)
        self.assertEqual(reference_torch.cuda.is_initialized(), initialized_before)

        if not reference_torch.cuda.is_available():
            self.skipTest("requires a visible CUDA accelerator")

        self.assertIs(reference_torch.cuda.is_available(), True)
        self.assertGreaterEqual(reference_torch.cuda.device_count(), 1)
        self.assertRegex(reference_torch.cuda.get_device_name(0), r"\S")
        self.assertIs(torch.cuda.is_available(), False)
        tensor = torch.tensor([1.0])
        self.assertEqual(tensor.device, torch.device("cpu"))
        self.assertIs(tensor.is_cuda, False)

    def test_callable_contract_matches_pytorch_2_13(self):
        actual = torch.cuda.is_available
        expected = reference_torch.cuda.is_available

        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"),
            expected.__module__,
        )
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(inspect.get_annotations(actual), inspect.get_annotations(expected))
        self.assertEqual(inspect.signature(actual), inspect.signature(expected))
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )
        self.assertIs(inspect.getmodule(actual), torch.cuda)
        self.assertIs(inspect.getmodule(expected), reference_torch.cuda)

    def test_pickling_matches_global_function_behavior(self):
        for module in (torch, reference_torch):
            function = module.cuda.is_available
            expected_path = function.__module__.encode()
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(module=module.__name__, protocol=protocol):
                    payload = pickle.dumps(function, protocol=protocol)
                    self.assertIn(expected_path, payload)
                    self.assertIs(pickle.loads(payload), function)

    def test_no_argument_errors_match_pytorch_2_13(self):
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
                lambda: torch.cuda.is_available(device=0),
                lambda: reference_torch.cuda.is_available(device=0),
            ),
            (
                lambda: torch.cuda.is_available(check_nvml=True),
                lambda: reference_torch.cuda.is_available(check_nvml=True),
            ),
            (
                lambda: torch.cuda.is_available(None, device=0),
                lambda: reference_torch.cuda.is_available(None, device=0),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
