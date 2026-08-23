import copy
import importlib
import inspect
import pickle
import pickletools
import sys
import typing
import unittest
from unittest import mock

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class DistributedGetDefaultBackendForDeviceReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "distributed.get_default_backend_for_device differentials "
                "require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def pickle_shape(self, function, protocol):
        shape = []
        for opcode, argument, _ in pickletools.genops(
            pickle.dumps(function, protocol=protocol)
        ):
            if opcode.name == "FRAME":
                argument = "<frame length>"
            elif isinstance(argument, str):
                argument = argument.replace("torch_rs", "torch")
            shape.append((opcode.name, argument))
        return shape

    def test_cpu_strings_indexed_strings_and_devices_match(self):
        values = (
            lambda module: "cpu",
            lambda module: "cpu:0",
            lambda module: "cpu:1",
            lambda module: "cpu:127",
            lambda module: "cpu:128",
            lambda module: "cpu:255",
            lambda module: "cpu:256",
            lambda module: "cpu:2147483647",
            lambda module: module.device("cpu"),
            lambda module: module.device("cpu", 0),
            lambda module: module.device("cpu:7"),
            lambda module: module.device(type="cpu", index=127),
        )

        for make_value in values:
            with self.subTest(value=make_value):
                actual = torch.distributed.get_default_backend_for_device(
                    make_value(torch)
                )
                expected = reference_torch.distributed.get_default_backend_for_device(
                    make_value(reference_torch)
                )
                self.assertIs(type(actual), type(expected))
                self.assertEqual(actual, expected)
                self.assertEqual(actual, "gloo")

        self.assertEqual(
            torch.distributed.get_default_backend_for_device(device="cpu:3"),
            reference_torch.distributed.get_default_backend_for_device(
                device="cpu:3"
            ),
        )

    def test_cpu_mapping_does_not_depend_on_gloo_availability(self):
        actual_distributed = torch.distributed
        expected_distributed = reference_torch.distributed
        actual_c10d = actual_distributed.distributed_c10d
        expected_c10d = expected_distributed.distributed_c10d

        with mock.patch.object(
            actual_c10d,
            "is_gloo_available",
            side_effect=AssertionError("backend probe"),
        ):
            with mock.patch.object(
                expected_c10d,
                "is_gloo_available",
                side_effect=AssertionError("backend probe"),
            ):
                self.assertEqual(
                    actual_distributed.get_default_backend_for_device("cpu:2"),
                    expected_distributed.get_default_backend_for_device("cpu:2"),
                )

        self.assertIs(actual_distributed.is_available(), False)
        self.assertIs(actual_distributed.is_gloo_available(), False)
        self.assertIs(type(expected_distributed.is_available()), bool)
        self.assertIs(type(expected_distributed.is_gloo_available()), bool)

    def test_signature_annotations_documentation_and_identity_match(self):
        actual_distributed = importlib.import_module("torch_rs.distributed")
        expected_distributed = importlib.import_module("torch.distributed")
        actual_c10d = importlib.import_module(
            "torch_rs.distributed.distributed_c10d"
        )
        expected_c10d = importlib.import_module(
            "torch.distributed.distributed_c10d"
        )
        actual = actual_distributed.get_default_backend_for_device
        expected = expected_distributed.get_default_backend_for_device

        self.assertIs(actual_c10d.get_default_backend_for_device, actual)
        self.assertIs(expected_c10d.get_default_backend_for_device, expected)
        self.assertEqual(
            str(inspect.signature(actual)).replace("torch_rs", "torch"),
            str(inspect.signature(expected)),
        )
        self.assertEqual(
            str(actual.__annotations__["device"]).replace("torch_rs", "torch"),
            str(expected.__annotations__["device"]),
        )
        self.assertIs(actual.__annotations__["return"], str)
        self.assertIs(expected.__annotations__["return"], str)
        self.assertEqual(
            str(typing.get_type_hints(actual)["device"]).replace(
                "torch_rs", "torch"
            ),
            str(typing.get_type_hints(expected)["device"]),
        )
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertIs(inspect.getmodule(actual), actual_c10d)
        self.assertIs(inspect.getmodule(expected), expected_c10d)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )

    def test_import_copy_wildcard_and_pickle_match(self):
        actual_distributed = torch.distributed
        expected_distributed = reference_torch.distributed
        actual_c10d = actual_distributed.distributed_c10d
        expected_c10d = expected_distributed.distributed_c10d
        actual = actual_distributed.get_default_backend_for_device
        expected = expected_distributed.get_default_backend_for_device

        self.assertIs(
            sys.modules["torch_rs.distributed.distributed_c10d"], actual_c10d
        )
        self.assertIs(
            sys.modules["torch.distributed.distributed_c10d"], expected_c10d
        )
        self.assertEqual(
            actual_c10d.__all__,
            [
                name
                for name in expected_c10d.__all__
                if name
                in {
                    "get_default_backend_for_device",
                    "get_pg_count",
                    "is_gloo_available",
                    "is_initialized",
                    "is_mpi_available",
                    "is_nccl_available",
                    "is_ucc_available",
                    "is_xccl_available",
                    "get_node_local_rank",
                }
            ],
        )

        for module, function in (
            (actual_distributed, actual),
            (expected_distributed, expected),
            (actual_c10d, actual),
            (expected_c10d, expected),
        ):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertIs(namespace["get_default_backend_for_device"], function)

        self.assertIs(copy.copy(actual), actual)
        self.assertIs(copy.copy(expected), expected)
        self.assertIs(copy.deepcopy(actual), actual)
        self.assertIs(copy.deepcopy(expected), expected)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(
                    pickle.loads(pickle.dumps(expected, protocol)), expected
                )
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

    def test_cpu_argument_errors_match(self):
        actual = torch.distributed.get_default_backend_for_device
        expected = reference_torch.distributed.get_default_backend_for_device
        cases = (
            (lambda: actual(), lambda: expected()),
            (lambda: actual("cpu", "cpu"), lambda: expected("cpu", "cpu")),
            (lambda: actual(value="cpu"), lambda: expected(value="cpu")),
            (lambda: actual(""), lambda: expected("")),
            (lambda: actual("cpu:-1"), lambda: expected("cpu:-1")),
            (lambda: actual("cpu:01"), lambda: expected("cpu:01")),
            (lambda: actual("cpu:"), lambda: expected("cpu:")),
            (
                lambda: actual("cpu:2147483648"),
                lambda: expected("cpu:2147483648"),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_non_cpu_mappings_and_execution_remain_intentionally_unsupported(self):
        actual_distributed = torch.distributed
        expected_distributed = reference_torch.distributed
        actual_c10d = actual_distributed.distributed_c10d

        expected_backends = {
            "cuda": "nccl",
            "cuda:0": "nccl",
            "mps": "gloo",
            "xpu": "xccl",
        }
        for specification, expected_backend in expected_backends.items():
            with self.subTest(specification=specification):
                self.assertEqual(
                    expected_distributed.get_default_backend_for_device(
                        specification
                    ),
                    expected_backend,
                )
                with self.assertRaises(RuntimeError):
                    actual_distributed.get_default_backend_for_device(
                        specification
                    )

        for name in (
            "Backend",
            "GroupMember",
            "ProcessGroup",
            "all_reduce",
            "destroy_process_group",
            "get_rank",
            "get_world_size",
            "init_process_group",
            "new_group",
        ):
            with self.subTest(name=name):
                self.assertTrue(hasattr(expected_distributed, name))
                self.assertFalse(hasattr(actual_distributed, name))
                self.assertFalse(hasattr(actual_c10d, name))


if __name__ == "__main__":
    unittest.main()
