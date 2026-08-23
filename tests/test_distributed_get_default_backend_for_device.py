import copy
import importlib
import inspect
import os
import pickle
import subprocess
import sys
import types
import typing
import unittest
from unittest import mock

import torch_rs as torch


FUNCTION_DOC = """
    Return the default backend for the given device.

    Args:
        device (Union[str, torch.device]): The device to get the default backend for.

    Returns:
        The default backend for the given device as a lower case string.

    """


class DistributedGetDefaultBackendForDeviceTests(unittest.TestCase):
    def test_cpu_strings_and_native_devices_return_gloo(self):
        function = torch.distributed.get_default_backend_for_device
        values = (
            "cpu",
            "cpu:0",
            "cpu:1",
            "cpu:127",
            "cpu:128",
            "cpu:255",
            "cpu:256",
            "cpu:2147483647",
            torch.device("cpu"),
            torch.device("cpu", 0),
            torch.device("cpu:7"),
            torch.device(type="cpu", index=127),
        )

        for value in values:
            with self.subTest(value=value):
                result = function(value)
                self.assertIs(type(result), str)
                self.assertEqual(result, "gloo")

        self.assertEqual(function(device="cpu:3"), "gloo")

    def test_cpu_mapping_does_not_check_backend_availability_or_host_state(self):
        distributed = torch.distributed
        distributed_c10d = distributed.distributed_c10d
        function = distributed.get_default_backend_for_device
        environments = (
            {},
            {"USE_DISTRIBUTED": "0", "USE_GLOO": "0"},
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "GLOO_SOCKET_IFNAME": "missing-interface",
                "MASTER_ADDR": "invalid.invalid",
                "MASTER_PORT": "not-a-port",
                "RANK": "invalid",
                "USE_DISTRIBUTED": "1",
                "USE_GLOO": "1",
                "WORLD_SIZE": "invalid",
            },
        )

        self.assertIs(distributed.is_available(), False)
        self.assertIs(distributed.is_gloo_available(), False)
        self.assertNotIn("is_available", function.__code__.co_names)
        self.assertNotIn("is_gloo_available", function.__code__.co_names)
        self.assertNotIn("Backend", function.__code__.co_names)
        self.assertFalse(hasattr(distributed_c10d, "Backend"))

        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    with mock.patch.object(
                        distributed,
                        "is_available",
                        side_effect=AssertionError("availability probe"),
                    ):
                        with mock.patch.object(
                            distributed_c10d,
                            "is_gloo_available",
                            side_effect=AssertionError("backend probe"),
                        ):
                            self.assertEqual(function("cpu:1"), "gloo")
                            self.assertEqual(
                                function(torch.device("cpu", 2)), "gloo"
                            )

    def test_signature_annotations_documentation_and_module_identity(self):
        distributed = importlib.import_module("torch_rs.distributed")
        distributed_c10d = importlib.import_module(
            "torch_rs.distributed.distributed_c10d"
        )
        function = distributed.get_default_backend_for_device

        self.assertIs(torch.distributed, distributed)
        self.assertIs(distributed.distributed_c10d, distributed_c10d)
        self.assertIs(
            distributed_c10d.get_default_backend_for_device, function
        )
        self.assertIs(sys.modules["torch_rs.distributed"], distributed)
        self.assertIs(
            sys.modules["torch_rs.distributed.distributed_c10d"],
            distributed_c10d,
        )
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(function)),
            "(device: str | torch_rs.device) -> str",
        )
        self.assertEqual(
            function.__annotations__,
            {"device": str | torch.device, "return": str},
        )
        self.assertEqual(
            typing.get_type_hints(function),
            {"device": str | torch.device, "return": str},
        )
        self.assertEqual(function.__name__, "get_default_backend_for_device")
        self.assertEqual(function.__qualname__, "get_default_backend_for_device")
        self.assertEqual(
            function.__module__, "torch_rs.distributed.distributed_c10d"
        )
        self.assertIs(inspect.getmodule(function), distributed_c10d)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_imports_copy_wildcards_and_pickle_use_the_canonical_module(self):
        distributed = torch.distributed
        distributed_c10d = distributed.distributed_c10d
        function = distributed.get_default_backend_for_device

        self.assertFalse(hasattr(distributed, "__all__"))
        self.assertEqual(
            distributed_c10d.__all__,
            [
                "get_default_backend_for_device",
                "get_pg_count",
                "is_gloo_available",
                "is_initialized",
                "is_mpi_available",
                "is_nccl_available",
                "is_ucc_available",
                "is_xccl_available",
                "get_node_local_rank",
            ],
        )

        direct_import = {}
        exec(
            "from torch_rs.distributed import get_default_backend_for_device",
            direct_import,
        )
        self.assertIs(direct_import["get_default_backend_for_device"], function)

        owner_import = {}
        exec(
            "from torch_rs.distributed.distributed_c10d import "
            "get_default_backend_for_device",
            owner_import,
        )
        self.assertIs(owner_import["get_default_backend_for_device"], function)

        distributed_namespace = {}
        exec("from torch_rs.distributed import *", distributed_namespace)
        self.assertIs(
            distributed_namespace["get_default_backend_for_device"], function
        )
        self.assertIs(
            distributed_namespace["distributed_c10d"], distributed_c10d
        )

        owner_namespace = {}
        exec(
            "from torch_rs.distributed.distributed_c10d import *",
            owner_namespace,
        )
        self.assertIs(
            owner_namespace["get_default_backend_for_device"], function
        )

        self.assertNotIn("get_default_backend_for_device", torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn(
            "get_default_backend_for_device", top_level_namespace
        )
        self.assertFalse(hasattr(torch, "get_default_backend_for_device"))

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(
                    b"torch_rs.distributed.distributed_c10d", payload
                )
                self.assertIs(pickle.loads(payload), function)

    def test_cpu_argument_errors_match_pytorch_2_13(self):
        function = torch.distributed.get_default_backend_for_device
        cases = (
            (
                lambda: function(),
                "get_default_backend_for_device() missing 1 required positional "
                "argument: 'device'",
            ),
            (
                lambda: function("cpu", "cpu"),
                "get_default_backend_for_device() takes 1 positional argument "
                "but 2 were given",
            ),
            (
                lambda: function(value="cpu"),
                "get_default_backend_for_device() got an unexpected keyword "
                "argument 'value'",
            ),
            (
                lambda: function(""),
                "Device string must not be empty",
            ),
            (
                lambda: function("cpu:-1"),
                "Invalid device string: 'cpu:-1'",
            ),
            (
                lambda: function("cpu:01"),
                "Invalid device string: 'cpu:01'",
            ),
            (
                lambda: function("cpu:"),
                "Invalid device string: 'cpu:'",
            ),
            (
                lambda: function("cpu:2147483648"),
                "Could not parse device index '2147483648' in device string "
                "'cpu:2147483648'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises((RuntimeError, TypeError)) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_non_cpu_mappings_and_distributed_execution_remain_unsupported(self):
        distributed = torch.distributed
        distributed_c10d = distributed.distributed_c10d
        function = distributed.get_default_backend_for_device

        for specification in ("cuda", "cuda:0", "mps", "xpu", "meta"):
            with self.subTest(specification=specification):
                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"device '{specification}' is not supported; only 'cpu' is implemented",
                ):
                    function(specification)

        for name in (
            "Backend",
            "GroupMember",
            "ProcessGroup",
            "ProcessGroupGloo",
            "all_reduce",
            "destroy_process_group",
            "get_rank",
            "get_world_size",
            "init_process_group",
            "new_group",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(distributed, name))
                self.assertFalse(hasattr(distributed_c10d, name))

    def test_importing_and_calling_does_not_import_pytorch(self):
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
    USE_DISTRIBUTED="1",
    USE_GLOO="0",
    MASTER_ADDR="invalid.invalid",
    MASTER_PORT="not-a-port",
    RANK="invalid",
    WORLD_SIZE="invalid",
    CUDA_VISIBLE_DEVICES="0",
)
import torch_rs as torch

function = torch.distributed.get_default_backend_for_device
assert function("cpu") == "gloo"
assert function("cpu:7") == "gloo"
assert function(torch.device("cpu", 3)) == "gloo"
assert torch.distributed.is_available() is False
assert torch.distributed.is_gloo_available() is False
assert not hasattr(torch.distributed, "init_process_group")
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
