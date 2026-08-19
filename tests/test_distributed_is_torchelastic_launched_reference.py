import copy
import importlib
import inspect
import os
import pickle
import pickletools
import sys
import types
import typing
import unittest
from unittest import mock

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class DistributedIsTorchelasticLaunchedReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "distributed.is_torchelastic_launched differentials require "
                "pinned PyTorch 2.13.0"
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

    def test_run_id_presence_semantics_match(self):
        actual = torch.distributed.is_torchelastic_launched
        expected = reference_torch.distributed.is_torchelastic_launched
        environments = (
            {},
            {"TORCHELASTIC_RESTART_COUNT": "0"},
            {
                "LOCAL_RANK": "0",
                "RANK": "0",
                "TORCHELASTIC_MAX_RESTARTS": "0",
                "TORCHELASTIC_RESTART_COUNT": "0",
                "WORLD_SIZE": "1",
            },
            {"TORCHELASTIC_RUN_ID": ""},
            {"TORCHELASTIC_RUN_ID": "0"},
            {"TORCHELASTIC_RUN_ID": "false"},
            {"TORCHELASTIC_RUN_ID": " "},
        )
        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    actual_result = actual()
                    expected_result = expected()
                self.assertIs(type(actual_result), bool)
                self.assertIs(type(expected_result), bool)
                self.assertIs(actual_result, expected_result)

        for value in (None, "", False, 0, object()):
            with self.subTest(getenv_result=value):
                with mock.patch.object(os, "getenv", return_value=value) as getenv:
                    actual_result = actual()
                    expected_result = expected()
                self.assertIs(actual_result, expected_result)
                self.assertEqual(
                    getenv.call_args_list,
                    [
                        mock.call("TORCHELASTIC_RUN_ID"),
                        mock.call("TORCHELASTIC_RUN_ID"),
                    ],
                )

    def test_signature_annotations_documentation_and_identity_match(self):
        actual_distributed = importlib.import_module("torch_rs.distributed")
        expected_distributed = importlib.import_module("torch.distributed")
        actual_c10d = importlib.import_module(
            "torch_rs.distributed.distributed_c10d"
        )
        expected_c10d = importlib.import_module(
            "torch.distributed.distributed_c10d"
        )
        actual = actual_distributed.is_torchelastic_launched
        expected = expected_distributed.is_torchelastic_launched

        self.assertIs(torch.distributed, actual_distributed)
        self.assertIs(reference_torch.distributed, expected_distributed)
        self.assertIs(actual_distributed.distributed_c10d, actual_c10d)
        self.assertIs(expected_distributed.distributed_c10d, expected_c10d)
        self.assertIs(actual_c10d.is_torchelastic_launched, actual)
        self.assertIs(expected_c10d.is_torchelastic_launched, expected)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual)), str(inspect.signature(expected))
        )
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(typing.get_type_hints(actual), typing.get_type_hints(expected))
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

    def test_imports_copy_wildcards_and_pickle_match(self):
        actual_distributed = torch.distributed
        expected_distributed = reference_torch.distributed
        actual_c10d = actual_distributed.distributed_c10d
        expected_c10d = expected_distributed.distributed_c10d
        actual = actual_distributed.is_torchelastic_launched
        expected = expected_distributed.is_torchelastic_launched
        supported_c10d = {
            "is_initialized",
            "is_nccl_available",
            "is_torchelastic_launched",
        }

        self.assertIs(
            sys.modules["torch_rs.distributed.distributed_c10d"], actual_c10d
        )
        self.assertIs(
            sys.modules["torch.distributed.distributed_c10d"], expected_c10d
        )
        self.assertEqual(
            hasattr(actual_distributed, "__all__"),
            hasattr(expected_distributed, "__all__"),
        )
        self.assertEqual(
            actual_c10d.__all__,
            [name for name in expected_c10d.__all__ if name in supported_c10d],
        )
        self.assertEqual(
            torch.__all__.count("distributed"),
            reference_torch.__all__.count("distributed"),
        )
        self.assertEqual(
            torch.__all__.count("is_torchelastic_launched"),
            reference_torch.__all__.count("is_torchelastic_launched"),
        )

        for module, function in (
            (actual_distributed, actual),
            (expected_distributed, expected),
            (actual_c10d, actual),
            (expected_c10d, expected),
        ):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertIs(namespace["is_torchelastic_launched"], function)

        actual_namespace = {}
        expected_namespace = {}
        exec("from torch_rs.distributed import *", actual_namespace)
        exec("from torch.distributed import *", expected_namespace)
        self.assertIs(actual_namespace["distributed_c10d"], actual_c10d)
        self.assertIs(expected_namespace["distributed_c10d"], expected_c10d)

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("distributed", namespace)
            self.assertNotIn("is_torchelastic_launched", namespace)

        self.assertIs(copy.copy(actual), actual)
        self.assertIs(copy.copy(expected), expected)
        self.assertIs(copy.deepcopy(actual), actual)
        self.assertIs(copy.deepcopy(expected), expected)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(pickle.loads(pickle.dumps(expected, protocol)), expected)
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.distributed.is_torchelastic_launched
        expected = reference_torch.distributed.is_torchelastic_launched
        cases = (
            (lambda: actual(None), lambda: expected(None)),
            (lambda: actual(None, None), lambda: expected(None, None)),
            (lambda: actual(enabled=True), lambda: expected(enabled=True)),
            (
                lambda: actual(None, enabled=True),
                lambda: expected(None, enabled=True),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_execution_and_elastic_launch_surfaces_remain_unsupported(self):
        actual_distributed = torch.distributed
        expected_distributed = reference_torch.distributed
        actual_c10d = actual_distributed.distributed_c10d
        expected_c10d = expected_distributed.distributed_c10d
        actual_public = {
            name for name in vars(actual_distributed) if not name.startswith("_")
        }
        expected_public = {
            name for name in vars(expected_distributed) if not name.startswith("_")
        }

        self.assertEqual(
            actual_public,
            {
                "distributed_c10d",
                "is_available",
                "is_initialized",
                "is_nccl_available",
                "is_torchelastic_launched",
            },
        )
        self.assertEqual(
            {
                name for name in vars(actual_c10d) if not name.startswith("_")
            },
            {
                "is_initialized",
                "is_nccl_available",
                "is_torchelastic_launched",
            },
        )
        unsupported = expected_public - actual_public
        self.assertTrue(unsupported)
        for name in unsupported:
            with self.subTest(name=name):
                self.assertFalse(hasattr(actual_distributed, name))

        for name in (
            "GroupMember",
            "ProcessGroup",
            "all_reduce",
            "destroy_process_group",
            "get_rank",
            "get_world_size",
            "init_process_group",
        ):
            with self.subTest(name=name):
                self.assertTrue(hasattr(expected_distributed, name))
                self.assertTrue(hasattr(expected_c10d, name))
                self.assertFalse(hasattr(actual_distributed, name))
                self.assertFalse(hasattr(actual_c10d, name))

        expected_elastic = importlib.import_module("torch.distributed.elastic")
        self.assertIsNotNone(expected_elastic)
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("torch_rs.distributed.elastic")

        with mock.patch.dict(
            os.environ, {"TORCHELASTIC_RUN_ID": ""}, clear=True
        ):
            self.assertIs(actual_distributed.is_torchelastic_launched(), True)
            self.assertIs(actual_distributed.is_available(), False)
            self.assertIs(actual_distributed.is_initialized(), False)


if __name__ == "__main__":
    unittest.main()
