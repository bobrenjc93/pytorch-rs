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
class GetDefaultDeviceReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "get_default_device differentials require pinned PyTorch 2.13.0"
            )

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

    def default_device_outcome(self, module):
        first = module.get_default_device()
        second = module.get_default_device()
        factories = (
            module.tensor([1.0, 2.0]),
            module.scalar_tensor(1.0),
            module.zeros((2, 0, 3)),
            module.ones((2, 3)),
            module.eye(2, 3),
            module.full((2,), 3.0),
        )
        return (
            str(first),
            repr(first),
            first.type,
            first.index,
            first == second,
            first is second,
            tuple(tensor.device == first for tensor in factories),
            tuple((tensor.device.type, tensor.device.index) for tensor in factories),
        )

    def threaded_outcome(self, module):
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = module.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    first = module.get_default_device()
                    second = module.get_default_device()
                    results[index] = (
                        str(first),
                        first.type,
                        first.index,
                        first == second,
                        first is second,
                        module.tensor(index).device == first,
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

    def cpu_setter_outcome(self, module):
        module.set_default_device(None)
        outcomes = []
        requests = (
            None,
            "cpu",
            module.device("cpu"),
            module.get_default_device(),
        )
        for requested in requests:
            result = module.set_default_device(requested)
            current = module.get_default_device()
            factories = (
                module.tensor([1.0, 2.0]),
                module.scalar_tensor(1.0),
                module.zeros((2, 0, 3)),
                module.ones((2, 3)),
                module.eye(2, 3),
                module.full((2,), 3.0),
            )
            outcomes.append(
                (
                    result,
                    str(current),
                    repr(current),
                    current.type,
                    current.index,
                    tuple(
                        tensor.device == module.device("cpu") for tensor in factories
                    ),
                    tuple(
                        (tensor.device.type, tensor.device.index)
                        for tensor in factories
                    ),
                )
            )
        return tuple(outcomes)

    def test_cpu_default_and_every_factory_match_pytorch_2_13(self):
        self.assertEqual(
            self.default_device_outcome(torch),
            self.default_device_outcome(reference_torch),
        )

    def test_cpu_setter_noop_forms_match_pytorch_2_13(self):
        self.assertEqual(
            self.cpu_setter_outcome(torch),
            self.cpu_setter_outcome(reference_torch),
        )

    def test_cpu_default_matches_when_cuda_is_visible(self):
        if not reference_torch.cuda.is_available():
            self.skipTest("requires a visible CUDA accelerator")

        self.assertGreaterEqual(reference_torch.cuda.device_count(), 1)
        self.assertEqual(torch.get_default_device(), torch.device("cpu"))
        self.assertEqual(
            reference_torch.get_default_device(),
            reference_torch.device("cpu"),
        )

    def test_grad_context_and_thread_stability_matches_pytorch_2_13(self):
        self.assertEqual(
            self.threaded_outcome(torch),
            self.threaded_outcome(reference_torch),
        )

        for module in (torch, reference_torch):
            with module.no_grad():
                first = module.get_default_device()
                self.assertEqual(first, module.device("cpu"))
                self.assertIsNot(module.get_default_device(), first)
                with module.no_grad():
                    self.assertEqual(
                        module.get_default_device(),
                        module.device("cpu"),
                    )

    def test_callable_metadata_matches_pytorch_2_13(self):
        actual = torch.get_default_device
        expected = reference_torch.get_default_device
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
        self.assertEqual(
            str(inspect.signature(actual)),
            str(inspect.signature(expected)),
        )
        self.assertEqual(
            "get_default_device" in torch.__all__,
            "get_default_device" in reference_torch.__all__,
        )
        self.assertEqual(torch.__all__.count("get_default_device"), 1)

        actual_setter = torch.set_default_device
        expected_setter = reference_torch.set_default_device
        self.assertIs(type(actual_setter), types.FunctionType)
        self.assertIs(type(expected_setter), types.FunctionType)
        self.assertEqual(actual_setter.__name__, expected_setter.__name__)
        self.assertEqual(actual_setter.__qualname__, expected_setter.__qualname__)
        self.assertEqual(actual_setter.__annotations__, expected_setter.__annotations__)
        self.assertEqual(
            actual_setter.__module__.replace("torch_rs", "torch"),
            expected_setter.__module__,
        )
        self.assertEqual(
            hasattr(actual_setter, "__text_signature__"),
            hasattr(expected_setter, "__text_signature__"),
        )
        self.assertEqual(
            str(inspect.signature(actual_setter)),
            str(inspect.signature(expected_setter)),
        )
        self.assertTrue(actual_setter.__doc__.startswith("Sets the default"))
        self.assertTrue(expected_setter.__doc__.startswith("Sets the default"))
        self.assertEqual(
            "set_default_device" in torch.__all__,
            "set_default_device" in reference_torch.__all__,
        )
        self.assertEqual(torch.__all__.count("set_default_device"), 1)

    def test_no_argument_errors_match_pytorch_2_13(self):
        cases = (
            (
                lambda: torch.get_default_device(None),
                lambda: reference_torch.get_default_device(None),
            ),
            (
                lambda: torch.get_default_device(None, None),
                lambda: reference_torch.get_default_device(None, None),
            ),
            (
                lambda: torch.get_default_device(device=None),
                lambda: reference_torch.get_default_device(device=None),
            ),
            (
                lambda: torch.get_default_device(foo=None),
                lambda: reference_torch.get_default_device(foo=None),
            ),
            (
                lambda: torch.get_default_device(None, device=None),
                lambda: reference_torch.get_default_device(None, device=None),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_set_default_device_argument_binding_errors_match_pytorch_2_13(self):
        cases = (
            (
                lambda: torch.set_default_device(),
                lambda: reference_torch.set_default_device(),
            ),
            (
                lambda: torch.set_default_device(None, None),
                lambda: reference_torch.set_default_device(None, None),
            ),
            (
                lambda: torch.set_default_device(value=None),
                lambda: reference_torch.set_default_device(value=None),
            ),
            (
                lambda: torch.set_default_device(None, device=None),
                lambda: reference_torch.set_default_device(None, device=None),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
