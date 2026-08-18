import inspect
import threading
import types
import unittest

import torch_rs as torch
from torch_rs.utils._device import DeviceContext


FUNCTION_DOC = """Sets the default ``torch.Tensor`` to be allocated on ``device``.  This
    does not affect factory function calls which are called with an explicit
    ``device`` argument.  Factory calls will be performed as if they
    were passed ``device`` as an argument.

    To only temporarily change the default device instead of setting it
    globally, use ``with torch.device(device):`` instead.

    The default device is initially ``cpu``.  If you set the default tensor
    device to another device (e.g., ``cuda``) without a device index, tensors
    will be allocated on whatever the current device for the device type,
    even after :func:`torch.cuda.set_device` is called.

    .. warning::

        This function imposes a slight performance cost on every Python
        call to the torch API (not just factory functions).  If this
        is causing problems for you, please comment on
        https://github.com/pytorch/pytorch/issues/92701

    .. note::

        This doesn't affect functions that create tensors that share the same memory as the input, like:
        :func:`torch.from_numpy` and :func:`torch.frombuffer`

    Args:
        device (device or string): the device to set as default

    Example::

        >>> # xdoctest: +SKIP("requires cuda, changes global state")
        >>> torch.get_default_device()
        device(type='cpu')
        >>> torch.set_default_device('cuda')  # current device is 0
        >>> torch.get_default_device()
        device(type='cuda', index=0)
        >>> torch.set_default_device('cuda')
        >>> torch.cuda.set_device('cuda:1')  # current device is 1
        >>> torch.get_default_device()
        device(type='cuda', index=1)
        >>> torch.set_default_device('cuda:1')
        >>> torch.get_default_device()
        device(type='cuda', index=1)

    """


class SetDefaultDeviceTests(unittest.TestCase):
    def setUp(self):
        torch.set_default_device(None)

    def tearDown(self):
        torch.set_default_device(None)

    def test_none_and_cpu_descriptors_update_the_default(self):
        cases = (
            (None, torch.device("cpu")),
            ("cpu", torch.device("cpu")),
            ("cpu:0", torch.device("cpu", 0)),
            ("cpu:3", torch.device("cpu", 3)),
            (torch.device("cpu"), torch.device("cpu")),
            (torch.device("cpu", 4), torch.device("cpu", 4)),
        )
        for value, expected in cases:
            with self.subTest(value=repr(value)):
                self.assertIsNone(torch.set_default_device(value))
                first = torch.get_default_device()
                second = torch.get_default_device()

                self.assertEqual(first, expected)
                self.assertEqual(second, expected)
                if expected.index is None:
                    self.assertIsNot(first, second)
                else:
                    self.assertIs(first, second)
                if isinstance(value, torch.device):
                    self.assertIsNot(first, value)

        torch.set_default_device("cpu:7")
        self.assertEqual(torch.get_default_device(), torch.device("cpu", 7))
        self.assertIsNone(torch.set_default_device(None))
        self.assertEqual(torch.get_default_device(), torch.device("cpu"))

    def test_factories_keep_unindexed_cpu_storage_and_explicit_devices_win(self):
        factories = (
            (
                "tensor",
                lambda: torch.tensor([1.0, 2.0]),
                lambda device: torch.tensor([1.0, 2.0], device=device),
            ),
            (
                "scalar_tensor",
                lambda: torch.scalar_tensor(1.0),
                lambda device: torch.scalar_tensor(1.0, device=device),
            ),
            (
                "zeros",
                lambda: torch.zeros((2, 0, 3)),
                lambda device: torch.zeros((2, 0, 3), device=device),
            ),
            (
                "ones",
                lambda: torch.ones((2, 3)),
                lambda device: torch.ones((2, 3), device=device),
            ),
            (
                "eye",
                lambda: torch.eye(2, 3),
                lambda device: torch.eye(2, 3, device=device),
            ),
            (
                "full",
                lambda: torch.full((2,), 3.0),
                lambda device: torch.full((2,), 3.0, device=device),
            ),
        )

        torch.set_default_device(torch.device("cpu", 6))
        self.assertEqual(torch.get_default_device(), torch.device("cpu", 6))
        for name, implicit_factory, explicit_factory in factories:
            with self.subTest(factory=name, argument="implicit"):
                tensor = implicit_factory()
                self.assertEqual(tensor.device, torch.device("cpu"))
                self.assertIsNone(tensor.device.index)
            with self.subTest(factory=name, argument="explicit"):
                tensor = explicit_factory(torch.device("cpu", 5))
                self.assertEqual(tensor.device, torch.device("cpu"))
                self.assertIsNone(tensor.device.index)
            with self.subTest(factory=name, argument="unsupported"):
                with self.assertRaisesRegex(RuntimeError, "only 'cpu' is implemented"):
                    explicit_factory("cuda:0")

        self.assertEqual(torch.get_default_device(), torch.device("cpu", 6))

    def test_device_context_stack_lifecycle_preserves_user_modes(self):
        stack = torch.overrides._get_current_function_mode_stack
        self.assertEqual(stack(), [])

        torch.set_default_device("cpu")
        first = stack()[0]
        self.assertIsInstance(first, DeviceContext)
        self.assertEqual(first.device, torch.device("cpu"))
        self.assertEqual(stack(), [first])

        torch.set_default_device("cpu:3")
        second = stack()[0]
        self.assertIsInstance(second, DeviceContext)
        self.assertIsNot(second, first)
        self.assertEqual(second.device, torch.device("cpu", 3))
        self.assertEqual(stack(), [second])

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return func(*args, **(kwargs or {}))

        mode = ForwardingMode()
        with mode:
            self.assertEqual(stack(), [second, mode])
            torch.set_default_device("cpu:4")
            replacement = stack()[0]
            self.assertIsInstance(replacement, DeviceContext)
            self.assertIsNot(replacement, second)
            self.assertEqual(stack(), [replacement, mode])
            torch.set_default_device(None)
            self.assertEqual(stack(), [mode])
        self.assertEqual(stack(), [])

        outer = DeviceContext("cpu:1")
        inner = DeviceContext("cpu:2")
        outer.__enter__()
        try:
            self.assertEqual(stack(), [outer])
            inner.__enter__()
            try:
                self.assertIs(inner.prev_mode, outer)
                self.assertEqual(stack(), [inner])
            finally:
                inner.__exit__(None, None, None)
            self.assertEqual(stack(), [outer])
        finally:
            outer.__exit__(None, None, None)
        self.assertEqual(stack(), [])

    def test_default_device_dispatches_supported_factories_through_modes(self):
        stack = torch.overrides._get_current_function_mode_stack
        factory_calls = (
            (torch.tensor, lambda: torch.tensor([1.0])),
            (torch.scalar_tensor, lambda: torch.scalar_tensor(1.0)),
            (torch.zeros, lambda: torch.zeros(1)),
            (torch.ones, lambda: torch.ones(1)),
            (torch.eye, lambda: torch.eye(1)),
            (torch.full, lambda: torch.full((1,), 2.0)),
        )
        factories = {function for function, _ in factory_calls}

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                if func in factories:
                    self.calls.append(
                        (
                            func,
                            types,
                            len(args),
                            None if kwargs is None else dict(kwargs),
                            tuple(type(mode).__name__ for mode in stack()),
                        )
                    )
                return func(*args, **(kwargs or {}))

        mode = RecordingMode()
        with mode:
            torch.set_default_device("cpu:6")
            mode.calls.clear()
            outputs = [call() for _, call in factory_calls]
            self.assertEqual(
                [type(active).__name__ for active in stack()],
                ["DeviceContext", "RecordingMode"],
            )
            torch.set_default_device(None)

        self.assertEqual(
            [call[0] for call in mode.calls],
            [function for function, _ in factory_calls],
        )
        self.assertTrue(all(call[1] == () for call in mode.calls))
        self.assertEqual([call[2] for call in mode.calls], [1, 1, 1, 1, 1, 2])
        self.assertTrue(all(call[3] is None for call in mode.calls))
        self.assertTrue(
            all(call[4] == ("DeviceContext",) for call in mode.calls)
        )
        for output in outputs:
            self.assertEqual(output.device, torch.device("cpu"))
            self.assertIsNone(output.device.index)
        self.assertEqual(stack(), [])

    def test_default_is_thread_local_and_is_not_inherited(self):
        torch.set_default_device("cpu:7")
        worker_count = 4
        ready = threading.Barrier(worker_count + 1)
        release = threading.Barrier(worker_count + 1)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                initial = torch.get_default_device()
                torch.set_default_device(f"cpu:{index}")
                first = torch.get_default_device()
                second = torch.get_default_device()
                ready.wait(timeout=10)
                release.wait(timeout=10)
                factory_device = torch.ones(1).device
                torch.set_default_device(None)
                reset = torch.get_default_device()
                results[index] = (
                    initial,
                    first,
                    second,
                    first is second,
                    factory_device,
                    reset,
                )
            except BaseException as error:
                errors.append(error)

        threads = [
            threading.Thread(target=worker, args=(index,))
            for index in range(worker_count)
        ]
        for thread in threads:
            thread.start()

        ready.wait(timeout=10)
        self.assertEqual(torch.get_default_device(), torch.device("cpu", 7))
        release.wait(timeout=10)
        for thread in threads:
            thread.join(timeout=10)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(torch.get_default_device(), torch.device("cpu", 7))
        for index, result in enumerate(results):
            initial, first, second, same_object, factory_device, reset = result
            self.assertEqual(initial, torch.device("cpu"))
            self.assertEqual(first, torch.device("cpu", index))
            self.assertEqual(second, first)
            self.assertTrue(same_object)
            self.assertEqual(factory_device, torch.device("cpu"))
            self.assertEqual(reset, torch.device("cpu"))

    def test_callable_metadata_and_export_match_pytorch_2_13(self):
        function = torch.set_default_device
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "set_default_device")
        self.assertEqual(function.__qualname__, "set_default_device")
        self.assertEqual(function.__module__, torch.__name__)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertEqual(
            function.__annotations__,
            {"device": "Device", "return": None},
        )
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertEqual(str(inspect.signature(function)), "(device: 'Device') -> None")
        self.assertEqual(torch.__all__.count("set_default_device"), 1)
        self.assertFalse(hasattr(torch._C, "set_default_device"))
        self.assertNotIn("set_default_device", torch._C.__all__)
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["set_default_device"], function)

    def test_binding_and_invalid_device_errors_preserve_the_default(self):
        torch.set_default_device("cpu:6")
        cases = (
            (
                lambda: torch.set_default_device(),
                TypeError,
                "set_default_device() missing 1 required positional argument: 'device'",
            ),
            (
                lambda: torch.set_default_device("cpu", "cpu"),
                TypeError,
                "set_default_device() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.set_default_device(foo="cpu"),
                TypeError,
                "set_default_device() got an unexpected keyword argument 'foo'",
            ),
            (
                lambda: torch.set_default_device("cpu", device="cpu"),
                TypeError,
                "set_default_device() got multiple values for argument 'device'",
            ),
            (
                lambda: torch.set_default_device(""),
                RuntimeError,
                "Device string must not be empty",
            ),
            (
                lambda: torch.set_default_device("cpu:01"),
                RuntimeError,
                "Invalid device string: 'cpu:01'",
            ),
            (
                lambda: torch.set_default_device("cpu:2147483648"),
                RuntimeError,
                "Could not parse device index '2147483648' in device string "
                "'cpu:2147483648'",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message):
                previous = torch.get_default_device()
                with self.assertRaises(error_type) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertIs(torch.get_default_device(), previous)

        with self.assertRaises(TypeError):
            torch.set_default_device(object())
        self.assertEqual(torch.get_default_device(), torch.device("cpu", 6))

    def test_accelerator_defaults_are_rejected_without_changing_state(self):
        torch.set_default_device("cpu:5")
        for device in ("cuda", "cuda:0", "meta", "mps"):
            with self.subTest(device=device):
                previous = torch.get_default_device()
                with self.assertRaisesRegex(RuntimeError, "only 'cpu' is implemented"):
                    torch.set_default_device(device)
                self.assertIs(torch.get_default_device(), previous)

        with self.assertRaises(TypeError):
            torch.set_default_device(0)
        self.assertEqual(torch.get_default_device(), torch.device("cpu", 5))


if __name__ == "__main__":
    unittest.main()
