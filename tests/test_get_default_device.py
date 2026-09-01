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
SET_FUNCTION_DOC = """Sets the default ``torch.Tensor`` to be allocated on ``device``.

    This CPU-only compatibility entrypoint is a no-op for ``None`` and the
    existing unindexed CPU default. Mutable default-device routing, CUDA or
    meta defaults, indexed CPU defaults, and ``torch.device(...)`` context
    behavior remain unsupported.
    """
UNSUPPORTED_DEFAULT_DEVICE = (
    "set_default_device(): only the unindexed CPU default is supported; "
    "mutable default-device routing, CUDA/meta defaults, indexed CPU "
    "defaults, and device context behavior are not implemented"
)


class DefaultDeviceTests(unittest.TestCase):
    def assert_default_cpu(self):
        first = torch.get_default_device()
        second = torch.get_default_device()

        self.assertIsInstance(first, torch.device)
        self.assertEqual(first, torch.device("cpu"))
        self.assertEqual(second, first)
        self.assertIsNot(second, first)
        self.assertEqual(first.type, "cpu")
        self.assertIsNone(first.index)
        return first

    def assert_factories_create_cpu_tensors(self):
        expected = torch.device("cpu")
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

    def test_getter_returns_a_fresh_unindexed_cpu_device_used_by_every_factory(self):
        self.assert_default_cpu()
        self.assert_factories_create_cpu_tensors()

    def test_setter_accepts_default_equivalent_cpu_forms_as_noops(self):
        accepted = (
            ("none", lambda: None),
            ("cpu string", lambda: "cpu"),
            ("cpu device", lambda: torch.device("cpu")),
            ("current default", torch.get_default_device),
            ("copied default", lambda: copy.copy(torch.get_default_device())),
            (
                "pickled default",
                lambda: pickle.loads(pickle.dumps(torch.get_default_device())),
            ),
        )
        for name, make_device in accepted:
            with self.subTest(form=name, call="positional"):
                self.assertIs(torch.set_default_device(make_device()), None)
                self.assert_default_cpu()
                self.assert_factories_create_cpu_tensors()
            with self.subTest(form=name, call="keyword"):
                self.assertIs(torch.set_default_device(device=make_device()), None)
                self.assert_default_cpu()
                self.assert_factories_create_cpu_tensors()

    def test_setter_rejects_default_changing_devices_without_mutating_factories(self):
        rejected = (
            ("indexed cpu string", lambda: "cpu:0"),
            ("indexed cpu descriptor", lambda: torch.device("cpu", 0)),
            ("another indexed cpu descriptor", lambda: torch.device("cpu:7")),
            ("cuda default", lambda: "cuda"),
            ("cuda indexed default", lambda: "cuda:0"),
            ("meta default", lambda: "meta"),
            ("meta indexed default", lambda: "meta:0"),
        )
        for name, make_device in rejected:
            with self.subTest(form=name):
                with self.assertRaises(NotImplementedError) as raised:
                    torch.set_default_device(make_device())
                self.assertEqual(str(raised.exception), UNSUPPORTED_DEFAULT_DEVICE)
                self.assertEqual(raised.exception.args, (UNSUPPORTED_DEFAULT_DEVICE,))
                self.assert_default_cpu()
                self.assert_factories_create_cpu_tensors()

    def test_setter_rejects_malformed_device_strings_before_mutating_state(self):
        cases = (
            ("", RuntimeError, "Device string must not be empty"),
            (
                "banana",
                RuntimeError,
                "Expected one of cpu, cuda, ipu, xpu, mkldnn, opengl, "
                "opencl, ideep, hip, ve, fpga, maia, xla, lazy, vulkan, "
                "mps, meta, hpu, mtia, privateuseone device type at start "
                "of device string: banana",
            ),
            (
                "CPU",
                RuntimeError,
                "Expected one of cpu, cuda, ipu, xpu, mkldnn, opengl, "
                "opencl, ideep, hip, ve, fpga, maia, xla, lazy, vulkan, "
                "mps, meta, hpu, mtia, privateuseone device type at start "
                "of device string: CPU",
            ),
            ("cpu:", RuntimeError, "Invalid device string: 'cpu:'"),
            ("cpu:01", RuntimeError, "Invalid device string: 'cpu:01'"),
            ("cpu:-1", RuntimeError, "Invalid device string: 'cpu:-1'"),
            (
                "cpu:2147483648",
                RuntimeError,
                "Could not parse device index '2147483648' in device string "
                "'cpu:2147483648'",
            ),
        )
        for value, error_type, message in cases:
            with self.subTest(device=value):
                with self.assertRaises(error_type) as raised:
                    torch.set_default_device(value)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assert_default_cpu()

        with self.assertRaises(TypeError):
            torch.set_default_device(0)
        self.assert_default_cpu()

    def test_result_is_stable_across_grad_contexts_and_threads(self):
        expected = torch.device("cpu")
        with torch.no_grad():
            first = torch.get_default_device()
            self.assertEqual(first, expected)
            self.assertIsNot(torch.get_default_device(), first)
            self.assertIs(torch.set_default_device(first), None)
            with torch.no_grad():
                self.assertEqual(torch.get_default_device(), expected)
                self.assertIs(torch.set_default_device("cpu"), None)
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
                    setter_result = torch.set_default_device(first)
                    second = torch.get_default_device()
                    results[index] = (
                        first,
                        setter_result,
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
            first, setter_result, second, same_object, tensor_device, grad_enabled = result
            self.assertEqual(first, expected)
            self.assertIs(setter_result, None)
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

    def test_setter_callable_metadata_matches_pytorch_2_13_signature_shape(self):
        function = torch.set_default_device
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "set_default_device")
        self.assertEqual(function.__qualname__, "set_default_device")
        self.assertEqual(function.__module__, torch.__name__)
        self.assertEqual(
            inspect.cleandoc(function.__doc__),
            inspect.cleandoc(SET_FUNCTION_DOC),
        )
        self.assertEqual(function.__annotations__, {"device": "Device", "return": None})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(str(inspect.signature(function)), "(device: 'Device') -> None")
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertEqual(torch.__all__.count("set_default_device"), 1)
        self.assertFalse(hasattr(torch._C, "set_default_device"))
        self.assertNotIn("set_default_device", torch._C.__all__)

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

    def test_setter_argument_binding_errors_match_pytorch_2_13(self):
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
                lambda: function(None, device=None),
                "set_default_device() got multiple values for argument 'device'",
            ),
            (
                lambda: function(foo=None),
                "set_default_device() got an unexpected keyword argument 'foo'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assert_default_cpu()

    def test_setter_imports_wildcard_copy_pickle_and_reload(self):
        package = importlib.import_module("torch_rs")
        function = package.set_default_device

        direct_import = {}
        exec("from torch_rs import set_default_device", direct_import)
        self.assertIs(direct_import["set_default_device"], function)

        namespace = {}
        exec("from torch_rs import *", namespace)
        self.assertIs(namespace["set_default_device"], function)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs", payload)
                self.assertIn(b"set_default_device", payload)
                self.assertIs(pickle.loads(payload), function)

        namespace_before_reload = package.__dict__
        reloaded = importlib.reload(package)
        self.assertIs(reloaded, package)
        self.assertIs(package.__dict__, namespace_before_reload)
        self.assertIsNot(package.set_default_device, function)
        self.assertIs(package.set_default_device("cpu"), None)
        self.assert_default_cpu()
        with self.assertRaises(pickle.PicklingError) as raised:
            pickle.dumps(function)
        self.assertIn(
            "it's not the same object as torch_rs.set_default_device",
            str(raised.exception),
        )

    def test_subprocess_default_device_state_is_isolated_and_cpu_only(self):
        script = r"""
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
assert torch.set_default_device(torch.get_default_device()) is None
for specification in ("cpu:0", "cuda", "meta"):
    try:
        torch.set_default_device(specification)
    except NotImplementedError:
        pass
    else:
        raise AssertionError(f"{specification!r} unexpectedly changed default")
assert torch.get_default_device() == torch.device("cpu")
assert torch.ones((1,)).device == torch.device("cpu")
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
        self.assert_default_cpu()


if __name__ == "__main__":
    unittest.main()
