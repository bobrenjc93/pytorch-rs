import copy
import importlib
import inspect
import pickle
import subprocess
import sys
import types
import unittest

import torch_rs as torch


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
    def assert_default_factories_are_cpu(self):
        expected = torch.device("cpu")
        factories = (
            torch.tensor([1.0, 2.0]),
            torch.scalar_tensor(1.0),
            torch.zeros((2, 0, 3)),
            torch.ones((2, 3)),
            torch.eye(2, 3),
            torch.full((2,), 3.0),
        )
        self.assertEqual(torch.get_default_device(), expected)
        for tensor in factories:
            with self.subTest(shape=tensor.shape):
                self.assertEqual(tensor.device, expected)
                self.assertEqual(tensor.device.type, "cpu")
                self.assertIsNone(tensor.device.index)

    def test_default_equivalent_cpu_requests_are_stateless_noops(self):
        class StringSubclass(str):
            pass

        values = (
            None,
            "cpu",
            StringSubclass("cpu"),
            torch.device("cpu"),
            torch.device(device=torch.device("cpu")),
        )
        for value in values:
            with self.subTest(value=repr(value)):
                self.assertIs(torch.set_default_device(value), None)
                self.assert_default_factories_are_cpu()

    def test_factory_metadata_stays_explicit_and_cpu_after_setter_calls(self):
        creators = (
            ("tensor", lambda **kw: torch.tensor(-2.5, **kw), (), -2.5),
            ("scalar_tensor", lambda **kw: torch.scalar_tensor(1.5, **kw), (), 1.5),
            ("zeros", lambda **kw: torch.zeros((2,), **kw), (2,), [0.0, 0.0]),
            ("ones", lambda **kw: torch.ones((2,), **kw), (2,), [1.0, 1.0]),
            (
                "eye",
                lambda **kw: torch.eye(2, 3, **kw),
                (2, 3),
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ),
            ("full", lambda **kw: torch.full((2,), -1.5, **kw), (2,), [-1.5, -1.5]),
        )
        for value in (None, "cpu", torch.device("cpu")):
            self.assertIs(torch.set_default_device(value), None)
            for name, create, shape, values in creators:
                with self.subTest(setter_value=repr(value), factory=name):
                    tensor = create(
                        dtype=torch.float32,
                        device=torch.device("cpu", 3),
                        requires_grad=True,
                    )
                    self.assertEqual(tensor.shape, shape)
                    self.assertEqual(tensor.tolist(), values)
                    self.assertIs(tensor.dtype, torch.float32)
                    self.assertEqual(tensor.device, torch.device("cpu"))
                    self.assertIs(tensor.requires_grad, True)

    def test_rejects_indexed_cpu_devices_without_changing_the_default(self):
        cases = (
            lambda: torch.set_default_device("cpu:0"),
            lambda: torch.set_default_device("cpu:1"),
            lambda: torch.set_default_device("cpu:255"),
            lambda: torch.set_default_device(torch.device("cpu", 0)),
            lambda: torch.set_default_device(torch.device("cpu", 1)),
        )
        for call in cases:
            with self.subTest(call=call):
                with self.assertRaises(NotImplementedError) as raised:
                    call()
                self.assertEqual(
                    str(raised.exception),
                    "set_default_device(): indexed CPU default devices are not supported",
                )
                self.assertEqual(torch.get_default_device(), torch.device("cpu"))

    def test_rejects_non_cpu_and_invalid_device_requests(self):
        cases = (
            lambda: torch.set_default_device("cuda"),
            lambda: torch.set_default_device("cuda:0"),
            lambda: torch.set_default_device("meta"),
            lambda: torch.set_default_device("mps"),
            lambda: torch.set_default_device(""),
            lambda: torch.set_default_device("cpu:01"),
            lambda: torch.set_default_device(True),
            lambda: torch.set_default_device(1),
            lambda: torch.set_default_device(1.5),
            lambda: torch.set_default_device(object()),
            lambda: torch.set_default_device(torch.float32),
            lambda: torch.set_default_device(torch.tensor(1.0)),
        )
        for call in cases:
            with self.subTest(call=call):
                with self.assertRaises((RuntimeError, TypeError, NotImplementedError)):
                    call()
                self.assertEqual(torch.get_default_device(), torch.device("cpu"))

    def test_validation_uses_native_device_class_not_rebound_public_names(self):
        original_device = torch.device
        original_get_default_device = torch.get_default_device
        torch.device = lambda *_args, **_kwargs: object()
        torch.get_default_device = lambda: object()
        try:
            self.assertIs(torch.set_default_device("cpu"), None)
        finally:
            torch.device = original_device
            torch.get_default_device = original_get_default_device

        self.assert_default_factories_are_cpu()

    def test_callable_metadata_matches_pytorch_2_13(self):
        package = importlib.import_module("torch_rs")
        function = package.set_default_device

        self.assertIs(torch, package)
        self.assertIs(sys.modules["torch_rs"], package)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "(device: 'Device') -> None")
        self.assertEqual(function.__annotations__, {"device": "Device", "return": None})
        self.assertEqual(function.__name__, "set_default_device")
        self.assertEqual(function.__qualname__, "set_default_device")
        self.assertEqual(function.__module__, "torch_rs")
        self.assertIs(inspect.getmodule(function), package)
        self.assertEqual(
            inspect.cleandoc(function.__doc__),
            inspect.cleandoc(FUNCTION_DOC),
        )
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertTrue(hasattr(torch, "set_default_device"))
        self.assertFalse(hasattr(torch._C, "_set_default_device"))

    def test_exports_copy_pickle_and_reload_use_the_canonical_module(self):
        package = importlib.import_module("torch_rs")
        function = package.set_default_device

        self.assertEqual(package.__all__.count("set_default_device"), 1)
        namespace = {}
        exec("from torch_rs import *", namespace)
        self.assertIs(namespace["set_default_device"], function)

        direct_import = {}
        exec("from torch_rs import set_default_device", direct_import)
        self.assertIs(direct_import["set_default_device"], function)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs", payload)
                self.assertIs(pickle.loads(payload), function)

        old_function = function
        self.assertIs(importlib.reload(package), package)
        self.assertIs(torch, package)
        self.assertIsNot(package.set_default_device, old_function)
        self.assertIs(old_function("cpu"), None)
        self.assertIs(package.set_default_device("cpu"), None)

    def test_binding_errors_match_pytorch_2_13(self):
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
                self.assertEqual(raised.exception.args, (message,))
                self.assertEqual(torch.get_default_device(), torch.device("cpu"))

        self.assertIs(function(device="cpu"), None)

    def test_subprocess_imports_without_pytorch_and_preserves_cpu_default(self):
        script = r"""
import copy
import importlib
import pickle
import sys


class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None


sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch
from torch_rs import set_default_device

namespace = {}
exec("from torch_rs import *", namespace)
function = torch.set_default_device

assert set_default_device is function
assert namespace["set_default_device"] is function
assert torch.__all__.count("set_default_device") == 1
assert not hasattr(torch._C, "_set_default_device")
assert function(None) is None
assert function("cpu") is None
assert function(torch.device("cpu")) is None
assert torch.get_default_device() == torch.device("cpu")
assert torch.zeros((1,), device=torch.device("cpu", 4)).device == torch.device("cpu")
assert copy.copy(function) is function
assert copy.deepcopy(function) is function
for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
    assert pickle.loads(pickle.dumps(function, protocol=protocol)) is function

old_function = function
assert importlib.reload(torch) is torch
assert torch.set_default_device is not old_function
assert old_function("cpu") is None
assert torch.set_default_device("cpu") is None
assert torch.get_default_device() == torch.device("cpu")
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
