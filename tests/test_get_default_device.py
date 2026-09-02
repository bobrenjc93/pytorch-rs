import copy
import contextlib
import inspect
import importlib
import pickle
import subprocess
import sys
import threading
import types
import unittest

import torch_rs as torch


FUNCTION_DOC = "Gets the default ``torch.Tensor`` to be allocated on ``device``"
SET_FUNCTION_DOC_PREFIX = (
    "Sets the default ``torch.Tensor`` to be allocated on ``device``."
)
UNSUPPORTED_DEFAULT_DEVICE_MESSAGE = (
    "set_default_device(): mutable default-device routing is not supported; "
    "only None and unindexed CPU are accepted"
)


class GetDefaultDeviceTests(unittest.TestCase):
    def assert_factory_outputs_are_cpu(self):
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
                self.assertEqual(tensor.device, torch.device("cpu"))
                self.assertEqual(tensor.device.type, "cpu")
                self.assertIsNone(tensor.device.index)

    def test_returns_a_fresh_unindexed_cpu_device_used_by_every_factory(self):
        first = torch.get_default_device()
        second = torch.get_default_device()

        self.assertIsInstance(first, torch.device)
        self.assertEqual(first, torch.device("cpu"))
        self.assertEqual(second, first)
        self.assertIsNot(second, first)
        self.assertEqual(first.type, "cpu")
        self.assertIsNone(first.index)
        self.assert_factory_outputs_are_cpu()

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

    def test_set_default_device_accepts_default_equivalent_cpu_forms_as_noops(self):
        requests = (
            ("none positional", lambda: torch.set_default_device(None)),
            ("none keyword", lambda: torch.set_default_device(device=None)),
            ("cpu string", lambda: torch.set_default_device("cpu")),
            ("cpu device", lambda: torch.set_default_device(torch.device("cpu"))),
            (
                "existing default",
                lambda: torch.set_default_device(torch.get_default_device()),
            ),
        )
        for name, call in requests:
            with self.subTest(name=name):
                before = torch.get_default_device()
                self.assertEqual(before, torch.device("cpu"))
                self.assertIsNone(call())
                after = torch.get_default_device()
                self.assertEqual(after, torch.device("cpu"))
                self.assertIsNone(after.index)
                self.assertIsNot(after, before)
                self.assert_factory_outputs_are_cpu()

    def test_set_default_device_rejects_non_default_devices_without_side_effects(self):
        unsupported = (
            "cuda",
            "cuda:0",
            "meta",
            "meta:0",
            "cpu:0",
            "cpu:1",
            torch.device("cpu", 0),
            torch.device("cpu", 1),
        )
        for requested in unsupported:
            with self.subTest(requested=repr(requested)):
                with self.assertRaises(NotImplementedError) as raised:
                    torch.set_default_device(requested)
                self.assertEqual(
                    str(raised.exception),
                    UNSUPPORTED_DEFAULT_DEVICE_MESSAGE,
                )
                self.assertEqual(
                    raised.exception.args,
                    (UNSUPPORTED_DEFAULT_DEVICE_MESSAGE,),
                )
                self.assertEqual(torch.get_default_device(), torch.device("cpu"))
                self.assert_factory_outputs_are_cpu()

    def test_set_default_device_rejects_invalid_types_without_side_effects(self):
        invalid = (
            (True, "bool"),
            (0, "int"),
            (1.0, "float"),
            (b"cpu", "bytes"),
            (object(), "object"),
            (torch.float32, "torch.dtype"),
            (torch.tensor(1.0), "Tensor"),
        )
        for requested, type_name in invalid:
            with self.subTest(type_name=type_name):
                message = (
                    "set_default_device(): argument 'device' must be "
                    f"torch.device, str, or None, not {type_name}"
                )
                with self.assertRaises(TypeError) as raised:
                    torch.set_default_device(requested)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertEqual(torch.get_default_device(), torch.device("cpu"))
                self.assert_factory_outputs_are_cpu()

    def test_set_default_device_callable_metadata_matches_pytorch_2_13(self):
        function = torch.set_default_device
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "set_default_device")
        self.assertEqual(function.__qualname__, "set_default_device")
        self.assertEqual(function.__module__, torch.__name__)
        self.assertTrue(function.__doc__.startswith(SET_FUNCTION_DOC_PREFIX))
        self.assertIn("CUDA/meta defaults", function.__doc__)
        self.assertEqual(
            function.__annotations__,
            {"device": "Device", "return": None},
        )
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(
            str(inspect.signature(function)),
            "(device: 'Device') -> None",
        )
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertEqual(torch.__all__.count("set_default_device"), 1)
        self.assertFalse(hasattr(torch._C, "_set_default_device"))

    def test_set_default_device_argument_binding_errors_match_pytorch_2_13(self):
        function = torch.set_default_device
        cases = (
            (
                lambda: function(),
                "set_default_device() missing 1 required positional argument: 'device'",
            ),
            (
                lambda: function(None, None),
                "set_default_device() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: function(value=None),
                "set_default_device() got an unexpected keyword argument 'value'",
            ),
            (
                lambda: function(None, device=None),
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

    def test_set_default_device_import_reload_copy_and_pickle_behavior(self):
        package = importlib.import_module("torch_rs")
        function = package.set_default_device
        getter = package.get_default_device

        self.assertIs(torch, package)
        self.assertIs(sys.modules["torch_rs"], package)
        self.assertEqual(package.__all__.count("set_default_device"), 1)
        namespace = {}
        exec("from torch_rs import *", namespace)
        self.assertIs(namespace["set_default_device"], function)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs", payload)
                self.assertIs(pickle.loads(payload), function)

        self.assertIs(importlib.reload(package), package)
        self.assertIsNot(package.set_default_device, function)
        self.assertIsNone(function(None))
        self.assertIsNone(package.set_default_device("cpu"))
        self.assertEqual(getter(), package.device("cpu"))
        self.assertEqual(package.get_default_device(), package.device("cpu"))

    def test_set_default_device_subprocess_isolation_and_no_pytorch_import(self):
        self.assertEqual(torch.get_default_device(), torch.device("cpu"))
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

assert torch.get_default_device() == torch.device("cpu")
assert torch.set_default_device(None) is None
assert torch.set_default_device("cpu") is None
assert torch.set_default_device(torch.device("cpu")) is None
assert torch.set_default_device(torch.get_default_device()) is None
assert torch.zeros((1,)).device == torch.device("cpu")
try:
    torch.set_default_device("meta")
except NotImplementedError:
    pass
else:
    raise AssertionError("meta default device must be rejected")
assert torch.get_default_device() == torch.device("cpu")
assert "set_default_device" in torch.__all__
namespace = {}
exec("from torch_rs import *", namespace)
assert namespace["set_default_device"] is torch.set_default_device
assert copy.copy(torch.set_default_device) is torch.set_default_device
assert pickle.loads(pickle.dumps(torch.set_default_device)) is torch.set_default_device
assert importlib.reload(torch) is torch
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
        self.assertEqual(torch.get_default_device(), torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
