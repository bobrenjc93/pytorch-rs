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
class SetDefaultDeviceReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "set_default_device differentials require pinned PyTorch 2.13.0"
            )

    def setUp(self):
        torch.set_default_device(None)
        reference_torch.set_default_device(None)

    def tearDown(self):
        torch.set_default_device(None)
        reference_torch.set_default_device(None)

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    @staticmethod
    def normalized_device(device):
        return str(device), repr(device), device.type, device.index

    def default_outcome(self, module, make_device):
        argument = make_device(module)
        result = module.set_default_device(argument)
        first = module.get_default_device()
        second = module.get_default_device()
        implicit = (
            module.tensor([1.0, 2.0]),
            module.scalar_tensor(1.0),
            module.zeros((2, 0, 3)),
            module.ones((2, 3)),
            module.eye(2, 3),
            module.full((2,), 3.0),
        )
        explicit = (
            module.tensor([1.0, 2.0], device="cpu:5"),
            module.scalar_tensor(1.0, device="cpu:5"),
            module.zeros((2, 0, 3), device="cpu:5"),
            module.ones((2, 3), device="cpu:5"),
            module.eye(2, 3, device="cpu:5"),
            module.full((2,), 3.0, device="cpu:5"),
        )
        outcome = (
            result is None,
            self.normalized_device(first),
            self.normalized_device(second),
            first is second,
            first is argument,
            tuple(self.normalized_device(tensor.device) for tensor in implicit),
            tuple(self.normalized_device(tensor.device) for tensor in explicit),
        )
        module.set_default_device(None)
        return outcome

    def test_cpu_defaults_and_factory_precedence_match_pytorch_2_13(self):
        cases = (
            lambda module: None,
            lambda module: "cpu",
            lambda module: "cpu:0",
            lambda module: "cpu:3",
            lambda module: module.device("cpu"),
            lambda module: module.device("cpu", 4),
        )
        for case, make_device in enumerate(cases):
            with self.subTest(case=case):
                self.assertEqual(
                    self.default_outcome(torch, make_device),
                    self.default_outcome(reference_torch, make_device),
                )

    def threaded_outcome(self, module):
        module.set_default_device("cpu:7")
        worker_count = 4
        ready = threading.Barrier(worker_count + 1)
        release = threading.Barrier(worker_count + 1)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                initial = module.get_default_device()
                module.set_default_device(f"cpu:{index}")
                first = module.get_default_device()
                second = module.get_default_device()
                ready.wait(timeout=10)
                release.wait(timeout=10)
                factory_device = module.ones(1).device
                module.set_default_device(None)
                reset = module.get_default_device()
                results[index] = (
                    self.normalized_device(initial),
                    self.normalized_device(first),
                    self.normalized_device(second),
                    first is second,
                    self.normalized_device(factory_device),
                    self.normalized_device(reset),
                )
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))

        threads = [
            threading.Thread(target=worker, args=(index,))
            for index in range(worker_count)
        ]
        for thread in threads:
            thread.start()

        ready.wait(timeout=10)
        main_during = self.normalized_device(module.get_default_device())
        release.wait(timeout=10)
        for thread in threads:
            thread.join(timeout=10)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        main_after = self.normalized_device(module.get_default_device())
        module.set_default_device(None)
        return results, main_during, main_after

    def test_thread_isolation_matches_pytorch_2_13(self):
        self.assertEqual(
            self.threaded_outcome(torch),
            self.threaded_outcome(reference_torch),
        )

    def test_callable_metadata_and_export_match_pytorch_2_13(self):
        actual = torch.set_default_device
        expected = reference_torch.set_default_device
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"),
            expected.__module__,
        )
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(str(inspect.signature(actual)), str(inspect.signature(expected)))
        self.assertEqual(
            "set_default_device" in torch.__all__,
            "set_default_device" in reference_torch.__all__,
        )
        self.assertEqual(torch.__all__.count("set_default_device"), 1)
        self.assertEqual(
            hasattr(torch._C, "set_default_device"),
            hasattr(reference_torch._C, "set_default_device"),
        )

    def test_binding_and_invalid_string_errors_match_pytorch_2_13(self):
        cases = (
            (
                lambda: torch.set_default_device(),
                lambda: reference_torch.set_default_device(),
            ),
            (
                lambda: torch.set_default_device("cpu", "cpu"),
                lambda: reference_torch.set_default_device("cpu", "cpu"),
            ),
            (
                lambda: torch.set_default_device(foo="cpu"),
                lambda: reference_torch.set_default_device(foo="cpu"),
            ),
            (
                lambda: torch.set_default_device("cpu", device="cpu"),
                lambda: reference_torch.set_default_device("cpu", device="cpu"),
            ),
            (
                lambda: torch.set_default_device(""),
                lambda: reference_torch.set_default_device(""),
            ),
            (
                lambda: torch.set_default_device("cpu:01"),
                lambda: reference_torch.set_default_device("cpu:01"),
            ),
            (
                lambda: torch.set_default_device("cpu:2147483648"),
                lambda: reference_torch.set_default_device("cpu:2147483648"),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                torch.set_default_device("cpu:6")
                reference_torch.set_default_device("cpu:6")
                self.assert_error_matches(actual_call, expected_call)
                self.assertEqual(
                    self.normalized_device(torch.get_default_device()),
                    self.normalized_device(reference_torch.get_default_device()),
                )

    def test_accelerator_rejection_is_an_explicit_cpu_only_difference(self):
        for device in ("cuda", "cuda:0", "meta", "mps"):
            with self.subTest(device=device):
                self.assertEqual(reference_torch.device(device).type, device.split(":")[0])
                torch.set_default_device("cpu:5")
                with self.assertRaisesRegex(RuntimeError, "only 'cpu' is implemented"):
                    torch.set_default_device(device)
                self.assertEqual(torch.get_default_device(), torch.device("cpu", 5))

        self.assertEqual(reference_torch.device(0).type, "cuda")
        with self.assertRaises(TypeError):
            torch.set_default_device(0)

    def test_cuda_reference_default_and_explicit_cpu_precedence(self):
        if not reference_torch.cuda.is_available():
            self.skipTest("requires a visible CUDA accelerator")

        reference_torch.set_default_device("cuda:0")
        try:
            implicit = reference_torch.ones(1)
            explicit = reference_torch.ones(1, device="cpu")
            self.assertEqual(implicit.device, reference_torch.device("cuda:0"))
            self.assertEqual(explicit.device, reference_torch.device("cpu"))
            self.assertEqual(implicit.item(), 1.0)
            self.assertEqual(explicit.item(), 1.0)
        finally:
            reference_torch.set_default_device(None)


if __name__ == "__main__":
    unittest.main()
