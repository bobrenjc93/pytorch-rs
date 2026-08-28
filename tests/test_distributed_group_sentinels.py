import copy
import importlib
import inspect
import pickle
import unittest

import torch_rs as torch


DEFAULT_GROUP_ERROR = (
    "Default process group has not been initialized, please make sure to "
    "call init_process_group."
)
INVALID_GROUP_ERROR = "Invalid process group specified"
WORLD_ASSIGNMENT_ERROR = (
    "torch_rs.distributed.WORLD does not support non-None process groups"
)
EXPECTED_C10D_ALL = [
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
]


class DistributedGroupSentinelTests(unittest.TestCase):
    def assert_error(self, error_type, message, call):
        with self.assertRaises(error_type) as raised:
            call()
        self.assertEqual(str(raised.exception), message)
        self.assertEqual(raised.exception.args, (message,))

    def test_classes_are_canonical_package_exports(self):
        distributed = torch.distributed
        distributed_c10d = importlib.import_module(
            "torch_rs.distributed.distributed_c10d"
        )
        group_member = distributed_c10d.GroupMember
        group = distributed_c10d.group

        self.assertIs(distributed.GroupMember, group_member)
        self.assertIs(distributed.group, group)
        self.assertIsNot(group_member, group)
        self.assertIs(type(group_member), type(group))
        self.assertEqual(type(group_member).__name__, "_WorldMeta")
        self.assertEqual(
            type(group_member).__module__,
            "torch_rs.distributed.distributed_c10d",
        )

        self.assertEqual(group_member.__name__, "GroupMember")
        self.assertEqual(group_member.__qualname__, "GroupMember")
        self.assertEqual(group_member.__doc__, "Group member class.")
        self.assertEqual(group.__name__, "group")
        self.assertEqual(group.__qualname__, "group")
        self.assertEqual(group.__doc__, "Group class. Placeholder.")
        self.assertEqual(group_member.__bases__, (object,))
        self.assertEqual(group.__bases__, (object,))
        self.assertEqual(str(inspect.signature(group_member)), "()")
        self.assertEqual(str(inspect.signature(group)), "()")

        self.assertEqual(group_member.NON_GROUP_MEMBER, -100)
        self.assertIs(group_member.WORLD, None)
        self.assertIs(group.WORLD, None)
        self.assertNotIn("WORLD", vars(group_member))
        self.assertNotIn("WORLD", vars(group))
        self.assertNotIn("NON_GROUP_MEMBER", vars(group))
        world_property = vars(type(group_member))["WORLD"]
        self.assertIsInstance(world_property, property)
        self.assertIsNotNone(world_property.fget)
        self.assertIsNotNone(world_property.fset)

        self.assertFalse(hasattr(distributed, "__all__"))
        self.assertEqual(distributed_c10d.__all__, EXPECTED_C10D_ALL)

        package_import = {}
        exec(
            "from torch_rs.distributed import GroupMember, group",
            package_import,
        )
        self.assertIs(package_import["GroupMember"], group_member)
        self.assertIs(package_import["group"], group)

        owner_import = {}
        exec(
            "from torch_rs.distributed.distributed_c10d import "
            "GroupMember, group",
            owner_import,
        )
        self.assertIs(owner_import["GroupMember"], group_member)
        self.assertIs(owner_import["group"], group)

        package_wildcard = {}
        exec("from torch_rs.distributed import *", package_wildcard)
        self.assertIs(package_wildcard["GroupMember"], group_member)
        self.assertIs(package_wildcard["group"], group)

        owner_wildcard = {}
        exec(
            "from torch_rs.distributed.distributed_c10d import *",
            owner_wildcard,
        )
        self.assertIs(owner_wildcard["GroupMember"], group_member)
        self.assertIs(owner_wildcard["group"], group)

    def test_copy_and_pickle_preserve_class_and_instance_behavior(self):
        distributed_c10d = torch.distributed.distributed_c10d

        for cls in (distributed_c10d.GroupMember, distributed_c10d.group):
            with self.subTest(cls=cls.__name__):
                self.assertIs(copy.copy(cls), cls)
                self.assertIs(copy.deepcopy(cls), cls)

                instance = cls()
                shallow = copy.copy(instance)
                deep = copy.deepcopy(instance)
                self.assertIsNot(shallow, instance)
                self.assertIsNot(deep, instance)
                self.assertIs(type(shallow), cls)
                self.assertIs(type(deep), cls)
                self.assertEqual(vars(shallow), {})
                self.assertEqual(vars(deep), {})

                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    with self.subTest(cls=cls.__name__, protocol=protocol):
                        self.assertIs(
                            pickle.loads(pickle.dumps(cls, protocol)), cls
                        )
                        restored = pickle.loads(
                            pickle.dumps(instance, protocol)
                        )
                        self.assertIs(type(restored), cls)
                        self.assertEqual(vars(restored), {})

    def test_sentinel_values_use_uninitialized_api_results_and_errors(self):
        distributed = torch.distributed
        group_member = distributed.GroupMember

        for world in (
            None,
            group_member.WORLD,
            distributed.group.WORLD,
        ):
            with self.subTest(world=world):
                self.assert_error(
                    ValueError,
                    DEFAULT_GROUP_ERROR,
                    lambda world=world: distributed.get_rank(world),
                )
                self.assert_error(
                    ValueError,
                    DEFAULT_GROUP_ERROR,
                    lambda world=world: distributed.get_world_size(world),
                )
                self.assert_error(
                    ValueError,
                    DEFAULT_GROUP_ERROR,
                    lambda world=world: distributed.get_backend(world),
                )
                self.assert_error(
                    ValueError,
                    DEFAULT_GROUP_ERROR,
                    lambda world=world: distributed.get_backend_config(world),
                )
                self.assert_error(
                    AssertionError,
                    "Process group cannot be None",
                    lambda world=world: distributed.destroy_process_group(world),
                )
                self.assertEqual(distributed.get_group_rank(world, 7), 7)
                self.assertEqual(distributed.get_global_rank(world, 7), 7)
                self.assert_error(
                    ValueError,
                    DEFAULT_GROUP_ERROR,
                    lambda world=world: distributed.get_process_group_ranks(world),
                )

        non_member = group_member.NON_GROUP_MEMBER
        self.assertEqual(distributed.get_rank(non_member), -1)
        self.assertEqual(distributed.get_world_size(non_member), -1)
        self.assert_error(
            ValueError,
            INVALID_GROUP_ERROR,
            lambda: distributed.get_backend(non_member),
        )
        self.assert_error(
            ValueError,
            INVALID_GROUP_ERROR,
            lambda: distributed.get_backend_config(non_member),
        )
        self.assertIsNone(distributed.destroy_process_group(non_member))
        self.assert_error(
            ValueError,
            "Group -100 is not registered, please create group with "
            "torch.distributed.new_group API",
            lambda: distributed.get_group_rank(non_member, 7),
        )
        self.assert_error(
            ValueError,
            "Group -100 is not registered, please create group with "
            "torch.distributed.new_group API",
            lambda: distributed.get_global_rank(non_member, 7),
        )
        with self.assertRaises(KeyError) as raised:
            distributed.get_process_group_ranks(non_member)
        self.assertEqual(raised.exception.args, (-100,))

        self.assertIs(distributed.is_available(), False)
        self.assertIs(distributed.is_initialized(), False)
        self.assertEqual(distributed.get_pg_count(), 0)
        for name in ("ProcessGroup", "init_process_group", "new_group", "all_reduce"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(distributed, name))
                self.assertFalse(
                    hasattr(distributed.distributed_c10d, name)
                )

    def test_reload_recreates_classes_and_rebinds_package_exports(self):
        distributed = torch.distributed
        distributed_c10d = distributed.distributed_c10d
        old_group_member = distributed_c10d.GroupMember
        old_group = distributed_c10d.group

        try:
            reloaded_c10d = importlib.reload(distributed_c10d)
            self.assertIs(reloaded_c10d, distributed_c10d)
            self.assertIsNot(reloaded_c10d.GroupMember, old_group_member)
            self.assertIsNot(reloaded_c10d.group, old_group)
            self.assertIs(distributed.GroupMember, old_group_member)
            self.assertIs(distributed.group, old_group)
            self.assertIs(reloaded_c10d.GroupMember.WORLD, None)
            self.assertIs(reloaded_c10d.group.WORLD, None)

            for stale_class in (old_group_member, old_group):
                with self.subTest(stale_class=stale_class.__name__):
                    self.assertIs(copy.copy(stale_class), stale_class)
                    with self.assertRaises(pickle.PicklingError):
                        pickle.dumps(stale_class)

            reloaded_distributed = importlib.reload(distributed)
            self.assertIs(reloaded_distributed, distributed)
            self.assertIs(
                reloaded_distributed.GroupMember,
                reloaded_c10d.GroupMember,
            )
            self.assertIs(reloaded_distributed.group, reloaded_c10d.group)
            self.assertIs(
                pickle.loads(pickle.dumps(reloaded_distributed.GroupMember)),
                reloaded_distributed.GroupMember,
            )
            self.assertIs(
                pickle.loads(pickle.dumps(reloaded_distributed.group)),
                reloaded_distributed.group,
            )
        finally:
            importlib.reload(distributed)

    def test_world_setter_rejects_non_none_groups_without_lifecycle_changes(self):
        distributed = torch.distributed
        distributed_c10d = distributed.distributed_c10d

        class UnreadableProcessGroup:
            def __getattribute__(self, name):
                if name.startswith("__"):
                    return object.__getattribute__(self, name)
                raise AssertionError(f"process group was inspected: {name}")

        candidate = UnreadableProcessGroup()
        for cls in (distributed_c10d.GroupMember, distributed_c10d.group):
            with self.subTest(cls=cls.__name__):
                self.assertIsNone(setattr(cls, "WORLD", None))
                self.assertIs(cls.WORLD, None)
                with self.assertRaises(NotImplementedError) as raised:
                    setattr(cls, "WORLD", candidate)
                self.assertEqual(str(raised.exception), WORLD_ASSIGNMENT_ERROR)
                self.assertEqual(
                    raised.exception.args,
                    (WORLD_ASSIGNMENT_ERROR,),
                )
                self.assertIs(distributed_c10d.GroupMember.WORLD, None)
                self.assertIs(distributed_c10d.group.WORLD, None)
                self.assertIs(distributed.is_initialized(), False)
                self.assertEqual(distributed.get_pg_count(), 0)
                self.assert_error(
                    ValueError,
                    DEFAULT_GROUP_ERROR,
                    distributed.get_rank,
                )
                self.assert_error(
                    ValueError,
                    DEFAULT_GROUP_ERROR,
                    distributed.get_world_size,
                )


if __name__ == "__main__":
    unittest.main()
