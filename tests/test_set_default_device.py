import copy
import importlib
import inspect
import pickle
import sys
import types
import unittest

import torch_rs as torch


FUNCTION_DOC = r"""Sets the default ``torch.Tensor`` to be allocated on ``device``.  This
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
    def assert_default_factories_are_cpu_float32(self):
        expected = torch.get_default_device()
        self.assertEqual(expected, torch.device("cpu"))
        self.assertEqual(expected.type, "cpu")
        self.assertIsNone(expected.index)
        self.assertIs(torch.get_default_dtype(), torch.float32)

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
                self.assertEqual(tensor.device, expected)
                self.assertEqual(tensor.device.type, "cpu")
                self.assertIsNone(tensor.device.index)
                self.assertIs(tensor.dtype, torch.float32)

    def test_unindexed_cpu_forms_are_stateless_noops(self):
        valid_devices = (
            "cpu",
            torch.device("cpu"),
            torch.device(type="cpu"),
            torch.device("cpu", index=None),
            torch.get_default_device(),
        )
        for value in valid_devices:
            with self.subTest(value=repr(value)):
                self.assertIsNone(torch.set_default_device(value))
                self.assert_default_factories_are_cpu_float32()

        self.assertIsNone(torch.set_default_device(device="cpu"))
        self.assert_default_factories_are_cpu_float32()

    def test_rebinding_public_device_does_not_change_validation(self):
        device_type = torch.device
        original_device = torch.device
        torch.device = lambda *_args, **_kwargs: object()
        try:
            self.assertIsNone(torch.set_default_device("cpu"))
            default_device = torch.get_default_device()
            self.assertIsInstance(default_device, device_type)
            self.assertEqual(default_device.type, "cpu")
            self.assertIsNone(default_device.index)
        finally:
            torch.device = original_device

        self.assert_default_factories_are_cpu_float32()

    def test_rejects_non_cpu_and_indexed_or_non_device_requests(self):
        indexed_message = (
            "set_default_device(): only the unindexed CPU device is supported"
        )
        cases = (
            (
                "cpu:0",
                NotImplementedError,
                indexed_message,
            ),
            (
                "cpu:1",
                NotImplementedError,
                indexed_message,
            ),
            (
                torch.device("cpu", 0),
                NotImplementedError,
                indexed_message,
            ),
            (
                torch.device(type="cpu", index=1),
                NotImplementedError,
                indexed_message,
            ),
            (
                "cuda",
                RuntimeError,
                "set_default_device(): device 'cuda' is not supported; only 'cpu' is implemented",
            ),
            (
                "cuda:0",
                RuntimeError,
                "set_default_device(): device 'cuda:0' is not supported; only 'cpu' is implemented",
            ),
            (
                "meta",
                RuntimeError,
                "set_default_device(): device 'meta' is not supported; only 'cpu' is implemented",
            ),
            ("", RuntimeError, "Device string must not be empty"),
            ("cpu:01", RuntimeError, "Invalid device string: 'cpu:01'"),
            (
                None,
                TypeError,
                "set_default_device(): argument 'device' must be torch.device or str, not NoneType",
            ),
            (
                object(),
                TypeError,
                "set_default_device(): argument 'device' must be torch.device or str, not object",
            ),
            (
                0,
                TypeError,
                "set_default_device(): argument 'device' must be torch.device or str, not int",
            ),
            (
                b"cpu",
                TypeError,
                "set_default_device(): argument 'device' must be torch.device or str, not bytes",
            ),
            (
                torch.float32,
                TypeError,
                "set_default_device(): argument 'device' must be torch.device or str, not dtype",
            ),
            (
                torch.tensor(1.0),
                TypeError,
                "set_default_device(): argument 'device' must be torch.device or str, not Tensor",
            ),
        )
        for value, error_type, message in cases:
            with self.subTest(value=repr(value)):
                with self.assertRaises(error_type) as raised:
                    torch.set_default_device(value)
                if message is not None:
                    self.assertEqual(str(raised.exception), message)
                self.assert_default_factories_are_cpu_float32()

    def test_callable_metadata_matches_pytorch_2_13(self):
        package = importlib.import_module("torch_rs")
        function = package.set_default_device

        self.assertIs(torch, package)
        self.assertIs(sys.modules["torch_rs"], package)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "set_default_device")
        self.assertEqual(function.__qualname__, "set_default_device")
        self.assertEqual(function.__module__, "torch_rs")
        self.assertIs(inspect.getmodule(function), package)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertEqual(function.__annotations__, {"device": "Device", "return": None})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

        signature = inspect.signature(function)
        self.assertEqual(str(signature), "(device: 'Device') -> None")
        self.assertEqual(
            signature.parameters["device"].kind,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        self.assertEqual(signature.parameters["device"].annotation, "Device")
        self.assertIsNone(signature.return_annotation)

    def test_imports_exports_copy_and_pickle_use_the_canonical_module(self):
        function = torch.set_default_device

        self.assertEqual(torch.__all__.count("set_default_device"), 1)
        self.assertFalse(hasattr(torch._C, "set_default_device"))

        direct_namespace = {}
        exec("from torch_rs import set_default_device", direct_namespace)
        self.assertIs(direct_namespace["set_default_device"], function)

        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["set_default_device"], function)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_reload_preserves_the_cpu_default_for_old_and_new_functions(self):
        package = importlib.import_module("torch_rs")
        old_setter = package.set_default_device
        old_getter = package.get_default_device

        self.assertIsNone(old_setter("cpu"))
        self.assertIs(importlib.reload(package), package)
        self.assertIs(torch, package)
        self.assertIsNot(package.set_default_device, old_setter)
        self.assertEqual(package.__all__.count("set_default_device"), 1)

        for setter in (old_setter, package.set_default_device):
            with self.subTest(setter=setter.__name__):
                self.assertIsNone(setter("cpu"))
                self.assertEqual(old_getter(), package.device("cpu"))
                self.assertEqual(package.get_default_device(), package.device("cpu"))
                self.assertIsNone(package.get_default_device().index)
                self.assertIs(package.get_default_dtype(), package.float32)

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
                self.assert_default_factories_are_cpu_float32()


if __name__ == "__main__":
    unittest.main()
