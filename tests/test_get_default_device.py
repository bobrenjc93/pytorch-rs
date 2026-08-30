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


FUNCTION_DOC = "Gets the default ``torch.Tensor`` to be allocated on ``device``"
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


class GetDefaultDeviceTests(unittest.TestCase):
    def setUp(self):
        torch.set_default_device(None)
        self.addCleanup(torch.set_default_device, None)

    def assert_default_device_and_factories_are_cpu(self, expected_default=None):
        if expected_default is None:
            expected_default = torch.device("cpu")
        else:
            expected_default = torch.device(expected_default)
        first = torch.get_default_device()
        second = torch.get_default_device()

        self.assertIsInstance(first, torch.device)
        self.assertEqual(first, expected_default)
        self.assertEqual(second, first)
        if first.index is None:
            self.assertIsNot(second, first)
        else:
            self.assertIs(second, first)
        self.assertEqual(first.type, "cpu")
        self.assertEqual(first.index, expected_default.index)

        factories = (
            ("tensor", lambda: torch.tensor([1.0, 2.0])),
            ("scalar_tensor", lambda: torch.scalar_tensor(1.0)),
            ("zeros", lambda: torch.zeros((2, 0, 3))),
            ("ones", lambda: torch.ones((2, 3))),
            ("eye", lambda: torch.eye(2, 3)),
            ("full", lambda: torch.full((2,), 3.0)),
            ("arange", lambda: torch.arange(3.0)),
        )
        for name, factory in factories:
            with self.subTest(factory=name):
                tensor = factory()
                self.assertEqual(tensor.device, torch.device("cpu"))
                self.assertEqual(tensor.device.type, "cpu")
                self.assertIsNone(tensor.device.index)
        self.assertIs(torch.get_default_dtype(), torch.float32)
        self.assertIs(torch.tensor([1.0]).dtype, torch.float32)
        self.assertEqual(torch.tensor([1.0]).type(), "torch.FloatTensor")

    def test_returns_a_fresh_unindexed_cpu_device_used_by_every_factory(self):
        self.assert_default_device_and_factories_are_cpu()

    def test_set_default_device_accepts_cpu_noops_and_reset(self):
        copied = copy.copy(torch.device("cpu"))
        pickled = pickle.loads(pickle.dumps(torch.device("cpu")))
        indexed_copy = copy.copy(torch.device("cpu:7"))
        indexed_pickle = pickle.loads(pickle.dumps(torch.device("cpu:127")))
        values = (
            None,
            "cpu",
            "cpu:0",
            "cpu:127",
            torch.device("cpu"),
            torch.device("cpu", None),
            torch.device("cpu:0"),
            torch.device("cpu", 7),
            copied,
            pickled,
            indexed_copy,
            indexed_pickle,
        )
        for value in values:
            with self.subTest(value=repr(value)):
                self.assertIs(torch.set_default_device(value), None)
                expected = torch.device("cpu") if value is None else torch.device(value)
                self.assert_default_device_and_factories_are_cpu(expected)

        self.assertIs(torch.set_default_device(device="cpu"), None)
        self.assert_default_device_and_factories_are_cpu()
        self.assertIs(torch.set_default_device(device="cpu:7"), None)
        self.assert_default_device_and_factories_are_cpu(torch.device("cpu", 7))
        self.assertIs(torch.set_default_device(device=None), None)
        self.assert_default_device_and_factories_are_cpu()

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

    def test_set_default_device_metadata_imports_copy_pickle_and_reload(self):
        function = torch.set_default_device
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "set_default_device")
        self.assertEqual(function.__qualname__, "set_default_device")
        self.assertEqual(function.__module__, torch.__name__)
        self.assertEqual(
            inspect.cleandoc(function.__doc__),
            inspect.cleandoc(SET_FUNCTION_DOC),
        )
        self.assertEqual(
            function.__annotations__,
            {"device": "Device", "return": None},
        )
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})

        signature = inspect.signature(function)
        self.assertEqual(str(signature), "(device: 'Device') -> None")
        self.assertEqual(
            signature.parameters["device"].kind,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        self.assertEqual(signature.parameters["device"].annotation, "Device")
        self.assertIsNone(signature.return_annotation)

        self.assertEqual(torch.__all__.count("set_default_device"), 1)
        self.assertFalse(hasattr(torch._C, "set_default_device"))
        self.assertNotIn("set_default_device", torch._C.__all__)

        direct_import = {}
        wildcard_import = {}
        exec("from torch_rs import set_default_device", direct_import)
        exec("from torch_rs import *", wildcard_import)
        self.assertIs(direct_import["set_default_device"], function)
        self.assertIs(wildcard_import["set_default_device"], function)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs", payload)
                self.assertIn(b"set_default_device", payload)
                self.assertIs(pickle.loads(payload), function)

        self.assertIsNone(function("cpu"))
        self.assertIs(importlib.reload(torch), torch)
        self.assertIsNone(function("cpu"))
        self.assertIsNone(function(None))
        self.assertIsNone(torch.set_default_device(torch.device("cpu")))
        self.assertIsNone(torch.set_default_device(torch.device("cpu:7")))
        self.assert_default_device_and_factories_are_cpu(torch.device("cpu", 7))
        self.assertIsNone(torch.set_default_device(None))
        self.assert_default_device_and_factories_are_cpu()

    def test_set_default_device_state_survives_package_reload_and_old_functions(self):
        old_getter = torch.get_default_device
        old_setter = torch.set_default_device

        self.assertIsNone(torch.set_default_device("cpu:7"))
        self.assert_default_device_and_factories_are_cpu(torch.device("cpu", 7))
        self.assertIs(importlib.reload(torch), torch)
        self.assert_default_device_and_factories_are_cpu(torch.device("cpu", 7))
        self.assertEqual(old_getter(), torch.device("cpu", 7))

        self.assertIsNone(old_setter(None))
        self.assert_default_device_and_factories_are_cpu()

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

    def test_set_default_device_rejects_non_cpu_requests(self):
        unsupported_values = (
            False,
            0,
            object(),
            [],
            {},
            torch.tensor(1.0),
            "cuda",
            "cuda:0",
            "meta",
        )
        for value in unsupported_values:
            with self.subTest(value=repr(value)):
                with self.assertRaises((TypeError, RuntimeError)):
                    torch.set_default_device(value)
                self.assert_default_device_and_factories_are_cpu()

        binding_errors = (
            (
                lambda: torch.set_default_device(),
                "set_default_device() missing 1 required positional argument: 'device'",
            ),
            (
                lambda: torch.set_default_device("cpu", "cpu"),
                "set_default_device() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.set_default_device(foo="cpu"),
                "set_default_device() got an unexpected keyword argument 'foo'",
            ),
            (
                lambda: torch.set_default_device("cpu", device="cpu"),
                "set_default_device() got multiple values for argument 'device'",
            ),
        )
        for call, message in binding_errors:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assert_default_device_and_factories_are_cpu()

    def test_set_default_device_does_not_import_or_enable_accelerators(self):
        script = r"""
import os
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
os.environ.update(
    CUDA_VISIBLE_DEVICES="0",
    PYTORCH_NVML_BASED_CUDA_CHECK="1",
)
import torch_rs

assert torch_rs.set_default_device("cpu") is None
assert torch_rs.get_default_device() == torch_rs.device("cpu")
assert torch_rs.ones(1).device == torch_rs.device("cpu")
assert torch_rs.tensor([1.0]).type() == "torch.FloatTensor"
assert not hasattr(torch_rs, "cuda")
assert "torch_rs.cuda" not in sys.modules
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
