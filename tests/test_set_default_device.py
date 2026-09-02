import copy
import importlib
import inspect
import pickle
import subprocess
import sys
import types
import unittest

import torch_rs as torch


FUNCTION_DOC = """
Sets the default ``torch.Tensor`` to be allocated on ``device``.  This
CPU-only compatibility entrypoint accepts only default-equivalent CPU
requests: ``None``, ``"cpu"``, and unindexed ``torch.device("cpu")``
values. It returns ``None`` and leaves factory output on the unindexed CPU
device. Mutable default-device routing, CUDA/meta defaults, indexed CPU
defaults, and ``with torch.device(...)`` device-context behavior remain
unsupported.
"""

INDEXED_CPU_ERROR = (
    "set_default_device(): indexed CPU default devices are not supported; "
    "only the unindexed CPU default is implemented"
)
UNSUPPORTED_DEVICE_ERROR = (
    "set_default_device(): device '{}' is not supported; "
    "only the unindexed CPU default is implemented"
)


class SetDefaultDeviceTests(unittest.TestCase):
    def assert_default_is_cpu(self):
        default = torch.get_default_device()
        self.assertEqual(default, torch.device("cpu"))
        self.assertEqual(default.type, "cpu")
        self.assertIsNone(default.index)

    def test_cpu_default_equivalent_requests_are_noops(self):
        copied_default = copy.copy(torch.get_default_device())
        pickled_default = pickle.loads(pickle.dumps(torch.get_default_device()))
        cases = (
            None,
            "cpu",
            torch.device("cpu"),
            torch.device("cpu", None),
            torch.get_default_device(),
            copied_default,
            pickled_default,
        )

        for requested_device in cases:
            with self.subTest(requested_device=repr(requested_device)):
                before = torch.get_default_device()
                self.assertIs(torch.set_default_device(requested_device), None)
                after = torch.get_default_device()
                self.assertEqual(after, before)
                self.assertIsNot(after, before)
                self.assert_default_is_cpu()

    def test_factory_outputs_remain_unindexed_cpu_after_every_noop_form(self):
        requests = (
            None,
            "cpu",
            torch.device("cpu"),
            torch.get_default_device(),
        )
        factories = (
            ("tensor", lambda: torch.tensor([1.0, 2.0])),
            ("scalar_tensor", lambda: torch.scalar_tensor(1.0)),
            ("zeros", lambda: torch.zeros((2, 0, 3))),
            ("ones", lambda: torch.ones((2, 3))),
            ("eye", lambda: torch.eye(2, 3)),
            ("full", lambda: torch.full((2,), 3.0)),
            ("arange", lambda: torch.arange(2.0)),
        )

        for requested_device in requests:
            with self.subTest(requested_device=repr(requested_device)):
                self.assertIs(torch.set_default_device(requested_device), None)
                self.assert_default_is_cpu()
                for name, factory in factories:
                    with self.subTest(factory=name):
                        tensor = factory()
                        self.assertEqual(tensor.device, torch.device("cpu"))
                        self.assertEqual(tensor.device.type, "cpu")
                        self.assertIsNone(tensor.device.index)

    def test_rejects_mutable_routing_and_non_cpu_default_requests(self):
        indexed_cpu_cases = (
            "cpu:0",
            "cpu:1",
            torch.device("cpu", 0),
            torch.device("cpu:127"),
        )
        for requested_device in indexed_cpu_cases:
            with self.subTest(requested_device=repr(requested_device)):
                with self.assertRaises(NotImplementedError) as raised:
                    torch.set_default_device(requested_device)
                self.assertEqual(str(raised.exception), INDEXED_CPU_ERROR)
                self.assert_default_is_cpu()

        non_cpu_cases = ("cuda", "cuda:0", "meta")
        for requested_device in non_cpu_cases:
            with self.subTest(requested_device=requested_device):
                with self.assertRaises(RuntimeError) as raised:
                    torch.set_default_device(requested_device)
                self.assertEqual(
                    str(raised.exception),
                    UNSUPPORTED_DEVICE_ERROR.format(requested_device),
                )
                self.assert_default_is_cpu()

    def test_rejects_invalid_argument_values_without_state_changes(self):
        class TensorLike:
            pass

        cases = (
            (
                lambda: torch.set_default_device(0),
                TypeError,
                "set_default_device(): argument 'device' must be torch.device, "
                "str, or None, not int",
            ),
            (
                lambda: torch.set_default_device(False),
                TypeError,
                "set_default_device(): argument 'device' must be torch.device, "
                "str, or None, not bool",
            ),
            (
                lambda: torch.set_default_device(TensorLike()),
                TypeError,
                "set_default_device(): argument 'device' must be torch.device, "
                "str, or None, not TensorLike",
            ),
            (
                lambda: torch.set_default_device(""),
                RuntimeError,
                "Device string must not be empty",
            ),
            (
                lambda: torch.set_default_device("cpu:"),
                RuntimeError,
                "Invalid device string: 'cpu:'",
            ),
            (
                lambda: torch.set_default_device("cpu:-1"),
                RuntimeError,
                "Invalid device string: 'cpu:-1'",
            ),
        )

        for call, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(error_type) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assert_default_is_cpu()

    def test_callable_metadata_matches_pytorch_2_13_shape(self):
        function = torch.set_default_device
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "set_default_device")
        self.assertEqual(function.__qualname__, "set_default_device")
        self.assertEqual(function.__module__, torch.__name__)
        self.assertEqual(inspect.cleandoc(function.__doc__), FUNCTION_DOC.strip())
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

    def test_binding_errors_match_pytorch_2_13(self):
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
                lambda: function(None, None, None),
                "set_default_device() takes 1 positional argument but 3 were given",
            ),
            (
                lambda: function(foo=None),
                "set_default_device() got an unexpected keyword argument 'foo'",
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
                self.assert_default_is_cpu()

    def test_imports_exports_copy_pickle_and_reload_use_the_canonical_function(self):
        function = torch.set_default_device

        direct_import = {}
        exec("from torch_rs import set_default_device", direct_import)
        self.assertIs(direct_import["set_default_device"], function)

        wildcard_import = {}
        exec("from torch_rs import *", wildcard_import)
        self.assertIs(wildcard_import["set_default_device"], function)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs", payload)
                self.assertIn(b"set_default_device", payload)
                self.assertIs(pickle.loads(payload), function)

        old_function = function
        namespace = torch.__dict__
        self.assertIs(importlib.reload(torch), torch)
        self.assertIs(torch.__dict__, namespace)
        self.assertIsNot(torch.set_default_device, old_function)
        self.assertIs(torch.set_default_device("cpu"), None)
        self.assert_default_is_cpu()
        self.assertIs(copy.copy(torch.set_default_device), torch.set_default_device)
        self.assertIs(
            copy.deepcopy(torch.set_default_device),
            torch.set_default_device,
        )
        self.assertIs(
            pickle.loads(pickle.dumps(torch.set_default_device)),
            torch.set_default_device,
        )
        with self.assertRaises(pickle.PicklingError) as raised:
            pickle.dumps(old_function)
        self.assertIn(
            "it's not the same object as torch_rs.set_default_device",
            str(raised.exception),
        )

    def test_subprocess_import_and_noop_are_isolated_from_pytorch_and_cuda(self):
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
import torch_rs as torch

assert torch.get_default_device() == torch.device("cpu")
assert torch.set_default_device(None) is None
assert torch.set_default_device("cpu") is None
assert torch.set_default_device(device=torch.get_default_device()) is None
assert torch.get_default_device() == torch.device("cpu")
try:
    torch.set_default_device("cuda")
except RuntimeError as error:
    assert "only the unindexed CPU default is implemented" in str(error)
else:
    raise AssertionError("CUDA default-device routing must remain unsupported")
assert torch.get_default_device() == torch.device("cpu")
assert torch.zeros((1,)).device == torch.device("cpu")
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
