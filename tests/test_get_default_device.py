import contextlib
import copy
import importlib
import inspect
import pickle
import subprocess
import sys
import threading
import types
import unittest

import torch_rs as torch


GET_FUNCTION_DOC = "Gets the default ``torch.Tensor`` to be allocated on ``device``"
SET_FUNCTION_DOC = """Sets the default ``torch.Tensor`` to be allocated on ``device``.  This
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


def assert_default_device(test_case, expected=None):
    if expected is None:
        expected = torch.device("cpu")

    default = torch.get_default_device()
    test_case.assertIsInstance(default, torch.device)
    test_case.assertEqual(default, expected)
    test_case.assertEqual(default.type, "cpu")
    test_case.assertEqual(default.index, expected.index)

    factories = (
        ("tensor", lambda: torch.tensor([1.0, 2.0])),
        ("scalar_tensor", lambda: torch.scalar_tensor(1.0)),
        ("zeros", lambda: torch.zeros((2, 0, 3))),
        ("ones", lambda: torch.ones((2, 3))),
        ("eye", lambda: torch.eye(2, 3)),
        ("full", lambda: torch.full((2,), 3.0)),
    )
    for name, factory in factories:
        with test_case.subTest(factory=name):
            tensor = factory()
            test_case.assertEqual(tensor.device, torch.device("cpu"))
            test_case.assertEqual(tensor.device.type, "cpu")
            test_case.assertIsNone(tensor.device.index)


class GetDefaultDeviceTests(unittest.TestCase):
    def setUp(self):
        torch.set_default_device(None)
        self.addCleanup(torch.set_default_device, None)

    def test_returns_a_fresh_unindexed_cpu_device_used_by_every_factory(self):
        first = torch.get_default_device()
        second = torch.get_default_device()

        self.assertIsInstance(first, torch.device)
        self.assertEqual(first, torch.device("cpu"))
        self.assertEqual(second, first)
        self.assertIsNot(second, first)
        self.assertEqual(first.type, "cpu")
        self.assertIsNone(first.index)

        assert_default_device(self)

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
        self.assertEqual(function.__doc__, GET_FUNCTION_DOC)
        self.assertEqual(function.__annotations__, {"return": "torch.device"})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(str(inspect.signature(function)), "() -> 'torch.device'")
        self.assertEqual(
            inspect.signature(function).return_annotation,
            "torch.device",
        )
        self.assertEqual(torch.__all__.count("get_default_device"), 1)
        self.assertTrue(hasattr(torch, "set_default_device"))
        self.assertEqual(torch.__all__.count("set_default_device"), 1)

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


class SetDefaultDeviceTests(unittest.TestCase):
    def setUp(self):
        torch.set_default_device(None)
        self.addCleanup(torch.set_default_device, None)

    def test_cpu_forms_update_reported_default_without_affecting_factories(self):
        class CpuString(str):
            def __eq__(self, other):
                raise AssertionError("set_default_device must not dispatch equality")

            def __str__(self):
                raise AssertionError("set_default_device must not dispatch str")

        supported = (
            (None, torch.device("cpu")),
            ("cpu", torch.device("cpu")),
            (CpuString("cpu"), torch.device("cpu")),
            ("cpu:0", torch.device("cpu", 0)),
            ("cpu:2", torch.device("cpu", 2)),
            (torch.device("cpu"), torch.device("cpu")),
            (torch.device("cpu", 0), torch.device("cpu", 0)),
            (torch.device("cpu:2"), torch.device("cpu", 2)),
            (copy.copy(torch.device("cpu:7")), torch.device("cpu", 7)),
            (
                pickle.loads(pickle.dumps(torch.device("cpu:127"))),
                torch.device("cpu", 127),
            ),
        )
        for value, expected in supported:
            with self.subTest(value=repr(value)):
                grad_enabled = torch.is_grad_enabled()
                self.assertIsNone(torch.set_default_device(value))
                self.assertIs(torch.is_grad_enabled(), grad_enabled)
                first = torch.get_default_device()
                second = torch.get_default_device()
                self.assertEqual(first, expected)
                self.assertEqual(second, expected)
                self.assertIsNot(first, second)
                assert_default_device(self, expected)

        with torch.no_grad():
            self.assertIsNone(torch.set_default_device("cpu:3"))
            self.assertFalse(torch.is_grad_enabled())
            assert_default_device(self, torch.device("cpu", 3))
        self.assertTrue(torch.is_grad_enabled())
        assert_default_device(self, torch.device("cpu", 3))

    def test_rejects_unsupported_devices_without_changing_defaults(self):
        invalid_values = (
            "cuda",
            "cuda:0",
            "mps",
            "xpu",
            "meta",
            "",
            "not a device",
            object(),
            1,
            True,
            b"cpu",
            torch.tensor(1.0),
            torch.float32,
        )
        expected = torch.tensor([1.0, 2.0]).tolist()
        torch.set_default_device("cpu:2")
        for value in invalid_values:
            with self.subTest(value=repr(value)):
                with self.assertRaises((TypeError, RuntimeError, NotImplementedError)):
                    torch.set_default_device(value)
                assert_default_device(self, torch.device("cpu", 2))
                self.assertEqual(torch.tensor([1.0, 2.0]).tolist(), expected)

    def test_rebinding_public_device_name_does_not_change_validation(self):
        original_device = torch.device
        torch.device = lambda specification: object()
        try:
            self.assertIsNone(torch.set_default_device("cpu"))
            default = torch.get_default_device()
            self.assertIsInstance(default, original_device)
            self.assertEqual(default, original_device("cpu"))

            self.assertIsNone(torch.set_default_device("cpu:0"))
            self.assertEqual(torch.get_default_device(), original_device("cpu:0"))

            with self.assertRaises(RuntimeError):
                torch.set_default_device("cuda")
            self.assertEqual(torch.get_default_device(), original_device("cpu:0"))
            self.assertEqual(torch.tensor([1.0]).device, original_device("cpu"))
        finally:
            torch.device = original_device

        assert_default_device(self, torch.device("cpu:0"))

    def test_default_device_contexts_and_non_cpu_allocation_remain_unsupported(self):
        with self.assertRaises(TypeError):
            with torch.device("cpu"):
                pass

        torch.set_default_device("cpu:2")
        for device in ("cuda", "mps", "xpu", "meta"):
            with self.subTest(device=device):
                with self.assertRaises(RuntimeError):
                    torch.zeros((1,), device=device)
                assert_default_device(self, torch.device("cpu", 2))

    def test_callable_metadata_matches_pytorch_2_13(self):
        function = torch.set_default_device
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "set_default_device")
        self.assertEqual(function.__qualname__, "set_default_device")
        self.assertEqual(function.__module__, torch.__name__)
        self.assertEqual(function.__doc__, SET_FUNCTION_DOC)
        self.assertEqual(function.__annotations__, {"device": "Device", "return": None})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})

        signature = inspect.signature(function)
        self.assertEqual(str(signature), "(device: 'Device') -> None")
        self.assertEqual(signature.parameters["device"].annotation, "Device")
        self.assertIsNone(signature.return_annotation)

        self.assertEqual(torch.__all__.count("set_default_device"), 1)
        namespace = {}
        exec("from torch_rs import *", namespace)
        self.assertIs(namespace["set_default_device"], function)
        self.assertFalse(hasattr(torch._C, "_set_default_device"))

    def test_reload_copy_and_pickle_use_the_canonical_function(self):
        package = importlib.import_module("torch_rs")
        self.assertIs(torch, package)
        self.assertIs(importlib.reload(package), package)

        function = package.set_default_device
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs", payload)
                self.assertIn(b"set_default_device", payload)
                self.assertIs(pickle.loads(payload), function)

        self.assertIsNone(function(None))
        self.assertIsNone(function(device="cpu:2"))
        assert_default_device(self, torch.device("cpu", 2))

    def test_argument_binding_errors_match_pytorch_2_13(self):
        function = torch.set_default_device
        cases = (
            (
                lambda: function(),
                "set_default_device() missing 1 required positional argument: 'device'",
            ),
            (
                lambda: function("cpu", "cpu"),
                "set_default_device() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: function(foo="cpu"),
                "set_default_device() got an unexpected keyword argument 'foo'",
            ),
            (
                lambda: function("cpu", device="cpu"),
                "set_default_device() got multiple values for argument 'device'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                assert_default_device(self)

        self.assertIsNone(function(device="cpu"))
        assert_default_device(self)

    def test_importing_calling_and_reloading_do_not_import_pytorch(self):
        script = r"""
import importlib
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

for device, expected in (
    (None, torch.device("cpu")),
    ("cpu", torch.device("cpu")),
    ("cpu:0", torch.device("cpu", 0)),
    ("cpu:2", torch.device("cpu", 2)),
    (torch.device("cpu"), torch.device("cpu")),
    (torch.device("cpu", 3), torch.device("cpu", 3)),
):
    assert torch.set_default_device(device) is None
    assert torch.get_default_device() == expected
    assert torch.zeros((1,)).device == torch.device("cpu")

torch.set_default_device("cpu:2")
for device in ("cuda", "mps", "xpu", "meta", object()):
    try:
        torch.set_default_device(device)
    except (TypeError, RuntimeError, NotImplementedError):
        pass
    else:
        raise AssertionError(f"unsupported device was accepted: {device!r}")
    assert torch.get_default_device() == torch.device("cpu", 2)

assert importlib.reload(torch) is torch
assert torch.set_default_device(device="cpu") is None
assert "set_default_device" in torch.__all__
assert not hasattr(torch._C, "_set_default_device")
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )


if __name__ == "__main__":
    unittest.main()
