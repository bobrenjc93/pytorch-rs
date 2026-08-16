import contextlib
import inspect
import threading
import types
import unittest

import torch_rs as torch


FUNCTION_DOC = "Gets the default ``torch.Tensor`` to be allocated on ``device``"


class GetDefaultDeviceTests(unittest.TestCase):
    def test_returns_a_fresh_unindexed_cpu_device_used_by_every_factory(self):
        first = torch.get_default_device()
        second = torch.get_default_device()

        self.assertIsInstance(first, torch.device)
        self.assertEqual(first, torch.device("cpu"))
        self.assertEqual(second, first)
        self.assertIsNot(second, first)
        self.assertEqual(first.type, "cpu")
        self.assertIsNone(first.index)

        factories = (
            ("tensor", lambda: torch.tensor([1.0, 2.0])),
            ("scalar_tensor", lambda: torch.scalar_tensor(1.0)),
            ("zeros", lambda: torch.zeros((2, 0, 3))),
            ("ones", lambda: torch.ones((2, 3))),
            ("eye", lambda: torch.eye(2, 3)),
            ("full", lambda: torch.full((2,), 3.0)),
        )
        for name, factory in factories:
            with self.subTest(factory=name):
                tensor = factory()
                self.assertEqual(tensor.device, first)
                self.assertEqual(tensor.device.type, "cpu")
                self.assertIsNone(tensor.device.index)

    def test_result_is_stable_across_grad_contexts_and_threads(self):
        expected = torch.device("cpu")
        with torch.no_grad():
            first = torch.get_default_device()
            self.assertEqual(first, expected)
            self.assertIsNot(torch.get_default_device(), first)
            with torch.no_grad():
                self.assertEqual(torch.get_default_device(), expected)
        self.assertEqual(torch.get_default_device(), expected)

        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = torch.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    first = torch.get_default_device()
                    second = torch.get_default_device()
                    results[index] = (
                        first,
                        second,
                        first is second,
                        torch.tensor(index).device,
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
            first, second, same_object, tensor_device, grad_enabled = result
            self.assertEqual(first, expected)
            self.assertEqual(second, expected)
            self.assertFalse(same_object)
            self.assertEqual(tensor_device, expected)
            self.assertEqual(grad_enabled, index % 2 == 0)

    def test_callable_metadata_matches_pytorch_2_13(self):
        function = torch.get_default_device
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "get_default_device")
        self.assertEqual(function.__qualname__, "get_default_device")
        self.assertEqual(function.__module__, torch.__name__)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertEqual(function.__annotations__, {"return": "torch.device"})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(str(inspect.signature(function)), "() -> 'torch.device'")
        self.assertEqual(
            inspect.signature(function).return_annotation,
            "torch.device",
        )
        self.assertEqual(torch.__all__.count("get_default_device"), 1)
        self.assertFalse(hasattr(torch, "set_default_device"))
        self.assertNotIn("set_default_device", torch.__all__)

    def test_rejects_all_arguments_with_pytorch_2_13_errors(self):
        function = torch.get_default_device
        cases = (
            (
                lambda: function(None),
                "get_default_device() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: function(None, None),
                "get_default_device() takes 0 positional arguments but 2 were given",
            ),
            (
                lambda: function(device=None),
                "get_default_device() got an unexpected keyword argument 'device'",
            ),
            (
                lambda: function(foo=None),
                "get_default_device() got an unexpected keyword argument 'foo'",
            ),
            (
                lambda: function(None, device=None),
                "get_default_device() got an unexpected keyword argument 'device'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)


if __name__ == "__main__":
    unittest.main()
