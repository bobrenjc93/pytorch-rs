import copy
import importlib
import inspect
import pickle
import pickletools
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class DistributedGroupSentinelReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "distributed group sentinel differentials require pinned "
                "PyTorch 2.13.0"
            )

    def outcome(self, call):
        try:
            result = call()
        except BaseException as error:
            return "error", type(error), str(error), error.args
        return "return", type(result), result

    def pickle_shape(self, value, protocol):
        shape = []
        for opcode, argument, _ in pickletools.genops(
            pickle.dumps(value, protocol=protocol)
        ):
            if opcode.name == "FRAME":
                argument = "<frame length>"
            elif isinstance(argument, str):
                argument = argument.replace("torch_rs", "torch")
            shape.append((opcode.name, argument))
        return shape

    def class_shape(self, cls):
        metaclass = type(cls)
        return {
            "metaclass": (
                metaclass.__module__.replace("torch_rs", "torch"),
                metaclass.__name__,
                metaclass.__qualname__,
            ),
            "module": cls.__module__.replace("torch_rs", "torch"),
            "name": cls.__name__,
            "qualname": cls.__qualname__,
            "doc": cls.__doc__,
            "bases": tuple(base.__name__ for base in cls.__bases__),
            "dict_keys": tuple(vars(cls)),
            "signature": str(inspect.signature(cls)),
            "public_dir": tuple(name for name in dir(cls) if not name.startswith("_")),
        }

    def test_class_metadata_exports_and_imports_match_pytorch_2_13(self):
        actual_distributed = torch.distributed
        expected_distributed = reference_torch.distributed
        actual_c10d = actual_distributed.distributed_c10d
        expected_c10d = expected_distributed.distributed_c10d

        supported = {
            "distributed_c10d",
            "is_available",
            "GroupMember",
            "destroy_process_group",
            "get_backend_config",
            "get_backend",
            "get_rank",
            "get_world_size",
            "get_pg_count",
            "group",
            "is_gloo_available",
            "is_initialized",
            "is_mpi_available",
            "is_nccl_available",
            "is_ucc_available",
            "is_xccl_available",
            "get_group_rank",
            "get_global_rank",
            "get_process_group_ranks",
            "get_node_local_rank",
        }
        self.assertEqual(
            actual_c10d.__all__,
            [
                name
                for name in expected_c10d.__all__
                if name in supported
            ],
        )

        for name in ("GroupMember", "group"):
            with self.subTest(name=name):
                actual = getattr(actual_c10d, name)
                expected = getattr(expected_c10d, name)
                self.assertIs(getattr(actual_distributed, name), actual)
                self.assertIs(getattr(expected_distributed, name), expected)
                self.assertEqual(self.class_shape(actual), self.class_shape(expected))

        self.assertIsNot(actual_c10d.GroupMember, actual_c10d.group)
        self.assertIs(
            type(actual_c10d.GroupMember), type(actual_c10d.group)
        )
        self.assertIs(
            type(expected_c10d.GroupMember), type(expected_c10d.group)
        )
        self.assertEqual(
            tuple(vars(type(actual_c10d.GroupMember))),
            tuple(vars(type(expected_c10d.GroupMember))),
        )
        actual_world = vars(type(actual_c10d.GroupMember))["WORLD"]
        expected_world = vars(type(expected_c10d.GroupMember))["WORLD"]
        self.assertIsInstance(actual_world, property)
        self.assertIsInstance(expected_world, property)
        self.assertEqual(
            str(inspect.signature(actual_world.fget)).replace("torch_rs", "torch"),
            str(inspect.signature(expected_world.fget)),
        )
        self.assertEqual(
            str(inspect.signature(actual_world.fset)).replace("torch_rs", "torch"),
            str(inspect.signature(expected_world.fset)),
        )

        self.assertEqual(actual_c10d.GroupMember.NON_GROUP_MEMBER, -100)
        self.assertEqual(
            actual_c10d.GroupMember.NON_GROUP_MEMBER,
            expected_c10d.GroupMember.NON_GROUP_MEMBER,
        )
        self.assertIs(actual_c10d.GroupMember.WORLD, None)
        self.assertIs(actual_c10d.group.WORLD, None)
        self.assertIs(expected_c10d.GroupMember.WORLD, None)
        self.assertIs(expected_c10d.group.WORLD, None)

        wildcard_orders = []
        for module, owner in (
            (actual_distributed, actual_c10d),
            (expected_distributed, expected_c10d),
        ):
            direct = {}
            exec(
                f"from {module.__name__} import GroupMember, group",
                direct,
            )
            self.assertIs(direct["GroupMember"], owner.GroupMember)
            self.assertIs(direct["group"], owner.group)

            wildcard = {}
            exec(f"from {module.__name__} import *", wildcard)
            self.assertIs(wildcard["GroupMember"], owner.GroupMember)
            self.assertIs(wildcard["group"], owner.group)
            wildcard_orders.append(
                tuple(name for name in wildcard if name in supported)
            )
        self.assertEqual(*wildcard_orders)

    def test_copy_and_pickle_behavior_matches_pytorch_2_13(self):
        actual_c10d = torch.distributed.distributed_c10d
        expected_c10d = reference_torch.distributed.distributed_c10d

        for name in ("GroupMember", "group"):
            actual = getattr(actual_c10d, name)
            expected = getattr(expected_c10d, name)
            with self.subTest(name=name):
                self.assertIs(copy.copy(actual), actual)
                self.assertIs(copy.copy(expected), expected)
                self.assertIs(copy.deepcopy(actual), actual)
                self.assertIs(copy.deepcopy(expected), expected)

                actual_instance = actual()
                expected_instance = expected()
                self.assertEqual(vars(actual_instance), vars(expected_instance))
                self.assertEqual(
                    type(copy.copy(actual_instance)).__name__,
                    type(copy.copy(expected_instance)).__name__,
                )
                self.assertEqual(
                    type(copy.deepcopy(actual_instance)).__name__,
                    type(copy.deepcopy(expected_instance)).__name__,
                )

                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    with self.subTest(name=name, protocol=protocol):
                        self.assertIs(
                            pickle.loads(pickle.dumps(actual, protocol)), actual
                        )
                        self.assertIs(
                            pickle.loads(pickle.dumps(expected, protocol)), expected
                        )
                        self.assertEqual(
                            self.pickle_shape(actual, protocol),
                            self.pickle_shape(expected, protocol),
                        )
                        self.assertEqual(
                            self.pickle_shape(actual_instance, protocol),
                            self.pickle_shape(expected_instance, protocol),
                        )

    def test_uninitialized_api_interoperation_matches_pytorch_2_13(self):
        actual_distributed = torch.distributed
        expected_distributed = reference_torch.distributed
        actual_c10d = actual_distributed.distributed_c10d
        expected_c10d = expected_distributed.distributed_c10d

        unary_names = (
            "get_rank",
            "get_world_size",
            "get_backend",
            "get_backend_config",
            "destroy_process_group",
            "get_process_group_ranks",
        )
        sentinel_pairs = (
            (actual_c10d.GroupMember.WORLD, expected_c10d.GroupMember.WORLD),
            (actual_c10d.group.WORLD, expected_c10d.group.WORLD),
            (
                actual_c10d.GroupMember.NON_GROUP_MEMBER,
                expected_c10d.GroupMember.NON_GROUP_MEMBER,
            ),
        )
        for name in unary_names:
            actual = getattr(actual_distributed, name)
            expected = getattr(expected_distributed, name)
            for actual_sentinel, expected_sentinel in sentinel_pairs:
                with self.subTest(name=name, sentinel=actual_sentinel):
                    self.assertEqual(
                        self.outcome(
                            lambda actual=actual, sentinel=actual_sentinel: actual(
                                sentinel
                            )
                        ),
                        self.outcome(
                            lambda expected=expected,
                            sentinel=expected_sentinel: expected(sentinel)
                        ),
                    )

        for name in ("get_group_rank", "get_global_rank"):
            actual = getattr(actual_distributed, name)
            expected = getattr(expected_distributed, name)
            for actual_sentinel, expected_sentinel in sentinel_pairs:
                with self.subTest(name=name, sentinel=actual_sentinel):
                    self.assertEqual(
                        self.outcome(
                            lambda actual=actual, sentinel=actual_sentinel: actual(
                                sentinel, 7
                            )
                        ),
                        self.outcome(
                            lambda expected=expected,
                            sentinel=expected_sentinel: expected(sentinel, 7)
                        ),
                    )

        self.assertIs(actual_distributed.is_available(), False)
        self.assertIs(actual_distributed.is_initialized(), False)
        self.assertEqual(actual_distributed.get_pg_count(), 0)
        for name in ("ProcessGroup", "init_process_group", "new_group", "all_reduce"):
            with self.subTest(unsupported=name):
                self.assertFalse(hasattr(actual_distributed, name))
                self.assertFalse(hasattr(actual_c10d, name))


if __name__ == "__main__":
    unittest.main()
