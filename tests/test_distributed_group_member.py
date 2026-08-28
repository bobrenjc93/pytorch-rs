import copy
import importlib
import inspect
import pickle
import subprocess
import sys
import unittest

import torch_rs as torch


DEFAULT_GROUP_ERROR = (
    "Default process group has not been initialized, please make sure to call "
    "init_process_group."
)
INVALID_GROUP_ERROR = "Invalid process group specified"
DESTROY_WORLD_ERROR = "Process group cannot be None"


class DistributedGroupMemberTests(unittest.TestCase):
    def assert_error(self, error_type, message, call):
        with self.assertRaises(error_type) as raised:
            call()
        self.assertEqual(str(raised.exception), message)
        self.assertEqual(raised.exception.args, (message,))

    def test_canonical_classes_and_uninitialized_values(self):
        distributed = torch.distributed
        distributed_c10d = distributed.distributed_c10d
        group_member = distributed.GroupMember
        group = distributed.group

        self.assertIs(group_member, distributed_c10d.GroupMember)
        self.assertIs(group, distributed_c10d.group)
        self.assertIsNot(group_member, group)
        self.assertIs(type(group_member), type(group))
        self.assertEqual(type(group_member).__name__, "_WorldMeta")
        self.assertEqual(
            type(group_member).__module__,
            "torch_rs.distributed.distributed_c10d",
        )

        self.assertIs(group_member.WORLD, None)
        self.assertIs(group.WORLD, None)
        self.assertIs(type(group_member.NON_GROUP_MEMBER), int)
        self.assertEqual(group_member.NON_GROUP_MEMBER, -100)
        self.assertFalse(hasattr(group, "NON_GROUP_MEMBER"))

        self.assertEqual(group_member.__name__, "GroupMember")
        self.assertEqual(group_member.__qualname__, "GroupMember")
        self.assertEqual(group_member.__doc__, "Group member class.")
        self.assertEqual(group.__name__, "group")
        self.assertEqual(group.__qualname__, "group")
        self.assertEqual(group.__doc__, "Group class. Placeholder.")
        self.assertEqual(inspect.signature(group_member), inspect.Signature())
        self.assertEqual(inspect.signature(group), inspect.Signature())
        self.assertEqual(
            list(group_member.__dict__),
            [
                "__module__",
                "__doc__",
                "NON_GROUP_MEMBER",
                "__dict__",
                "__weakref__",
            ],
        )
        self.assertEqual(
            list(group.__dict__),
            ["__module__", "__doc__", "__dict__", "__weakref__"],
        )
        self.assertEqual(group_member.__annotations__, {})
        self.assertEqual(group.__annotations__, {})

        for cls in (group_member, group):
            instance = cls()
            self.assertIs(type(instance), cls)
            self.assertEqual(instance.__dict__, {})
            self.assertFalse(hasattr(instance, "WORLD"))
        self.assertEqual(group_member().NON_GROUP_MEMBER, -100)
        self.assertFalse(hasattr(group(), "NON_GROUP_MEMBER"))

    def test_world_is_a_shared_class_property(self):
        distributed_c10d = torch.distributed.distributed_c10d
        group_member = distributed_c10d.GroupMember
        group = distributed_c10d.group
        world_meta = type(group_member)
        world_property = world_meta.__dict__["WORLD"]

        self.assertIsInstance(world_property, property)
        self.assertIsNone(world_property.__doc__)
        self.assertEqual(world_property.fget.__name__, "WORLD")
        self.assertEqual(world_property.fget.__qualname__, "_WorldMeta.WORLD")
        self.assertEqual(world_property.fset.__name__, "WORLD")
        self.assertEqual(world_property.fset.__qualname__, "_WorldMeta.WORLD")
        self.assertEqual(
            str(inspect.signature(world_property.fget)),
            "(cls) -> torch_rs.distributed.distributed_c10d.ProcessGroup | None",
        )
        self.assertEqual(
            str(inspect.signature(world_property.fset)),
            "(cls, pg: torch_rs.distributed.distributed_c10d.ProcessGroup | None)",
        )

        first = object()
        second = object()
        try:
            group_member.WORLD = first
            self.assertIs(group_member.WORLD, first)
            self.assertIs(group.WORLD, first)
            group.WORLD = second
            self.assertIs(group_member.WORLD, second)
            self.assertIs(group.WORLD, second)
        finally:
            group_member.WORLD = None
        self.assertIs(group_member.WORLD, None)
        self.assertIs(group.WORLD, None)

    def test_imports_copy_and_pickle_are_canonical(self):
        distributed = torch.distributed
        distributed_c10d = distributed.distributed_c10d

        direct = {}
        exec("from torch_rs.distributed import GroupMember, group", direct)
        self.assertIs(direct["GroupMember"], distributed_c10d.GroupMember)
        self.assertIs(direct["group"], distributed_c10d.group)

        owner = {}
        exec(
            "from torch_rs.distributed.distributed_c10d import GroupMember, group",
            owner,
        )
        self.assertIs(owner["GroupMember"], distributed_c10d.GroupMember)
        self.assertIs(owner["group"], distributed_c10d.group)

        for module_name in (
            "torch_rs.distributed",
            "torch_rs.distributed.distributed_c10d",
        ):
            namespace = {}
            exec(f"from {module_name} import *", namespace)
            self.assertIs(namespace["GroupMember"], distributed_c10d.GroupMember)
            self.assertIs(namespace["group"], distributed_c10d.group)

        self.assertNotIn("GroupMember", torch.__all__)
        self.assertNotIn("group", torch.__all__)
        self.assertFalse(hasattr(torch, "GroupMember"))
        self.assertFalse(hasattr(torch, "group"))

        for cls in (distributed_c10d.GroupMember, distributed_c10d.group):
            self.assertIs(copy.copy(cls), cls)
            self.assertIs(copy.deepcopy(cls), cls)
            instance = cls()
            self.assertIsNot(copy.copy(instance), instance)
            self.assertIs(type(copy.copy(instance)), cls)
            self.assertIsNot(copy.deepcopy(instance), instance)
            self.assertIs(type(copy.deepcopy(instance)), cls)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(cls=cls.__name__, protocol=protocol):
                    class_payload = pickle.dumps(cls, protocol=protocol)
                    self.assertIn(
                        b"torch_rs.distributed.distributed_c10d", class_payload
                    )
                    self.assertIs(pickle.loads(class_payload), cls)
                    restored = pickle.loads(
                        pickle.dumps(instance, protocol=protocol)
                    )
                    self.assertIs(type(restored), cls)
                    self.assertEqual(restored.__dict__, {})

    def test_reload_recreates_classes_and_resets_world(self):
        distributed = torch.distributed
        distributed_c10d = distributed.distributed_c10d

        for _ in range(3):
            old_group_member = distributed.GroupMember
            old_group = distributed.group
            old_meta = type(old_group_member)
            old_world = distributed_c10d._world
            old_group_member.WORLD = object()

            distributed_c10d = importlib.reload(distributed_c10d)
            self.assertIs(distributed.GroupMember, old_group_member)
            self.assertIs(distributed.group, old_group)
            self.assertIsNot(distributed_c10d.GroupMember, old_group_member)
            self.assertIsNot(distributed_c10d.group, old_group)
            self.assertIsNot(type(distributed_c10d.GroupMember), old_meta)
            self.assertIsNot(distributed_c10d._world, old_world)
            self.assertIs(old_group_member.WORLD, None)
            self.assertIs(old_group.WORLD, None)
            self.assertIs(distributed_c10d.GroupMember.WORLD, None)
            self.assertIs(distributed_c10d.group.WORLD, None)
            with self.assertRaises(pickle.PicklingError):
                pickle.dumps(old_group_member)
            with self.assertRaises(pickle.PicklingError):
                pickle.dumps(old_group)

            distributed = importlib.reload(distributed)
            self.assertIs(torch.distributed, distributed)
            self.assertIs(distributed.distributed_c10d, distributed_c10d)
            self.assertIs(distributed.GroupMember, distributed_c10d.GroupMember)
            self.assertIs(distributed.group, distributed_c10d.group)

    def test_sentinels_interoperate_with_uninitialized_apis(self):
        distributed = torch.distributed
        worlds = (distributed.GroupMember.WORLD, distributed.group.WORLD)
        non_member = distributed.GroupMember.NON_GROUP_MEMBER

        for world in worlds:
            with self.subTest(world=world):
                self.assert_error(
                    AssertionError,
                    DESTROY_WORLD_ERROR,
                    lambda world=world: distributed.destroy_process_group(world),
                )
                for function in (
                    distributed.get_backend_config,
                    distributed.get_backend,
                    distributed.get_rank,
                    distributed.get_world_size,
                ):
                    self.assert_error(
                        ValueError,
                        DEFAULT_GROUP_ERROR,
                        lambda function=function, world=world: function(world),
                    )

        self.assertIsNone(distributed.destroy_process_group(non_member))
        rank = distributed.get_rank(non_member)
        world_size = distributed.get_world_size(non_member)
        self.assertIs(type(rank), int)
        self.assertIs(type(world_size), int)
        self.assertEqual(rank, -1)
        self.assertEqual(world_size, -1)
        for function in (
            distributed.get_backend_config,
            distributed.get_backend,
        ):
            self.assert_error(
                ValueError,
                INVALID_GROUP_ERROR,
                lambda function=function: function(non_member),
            )

        self.assertEqual(
            distributed.get_group_rank(distributed.GroupMember.WORLD, 7), 7
        )
        self.assertEqual(
            distributed.get_global_rank(distributed.group.WORLD, 11), 11
        )
        with self.assertRaisesRegex(ValueError, "^Group -100 is not registered"):
            distributed.get_group_rank(non_member, 0)
        with self.assertRaisesRegex(ValueError, "^Group -100 is not registered"):
            distributed.get_global_rank(non_member, 0)
        with self.assertRaises(KeyError) as raised:
            distributed.get_process_group_ranks(non_member)
        self.assertEqual(raised.exception.args, (-100,))

        self.assertIs(distributed.is_available(), False)
        self.assertIs(distributed.is_initialized(), False)
        self.assertEqual(distributed.get_pg_count(), 0)
        for name in ("ProcessGroup", "init_process_group", "new_group", "all_reduce"):
            self.assertFalse(hasattr(distributed, name))
            self.assertFalse(hasattr(distributed.distributed_c10d, name))

    def test_importing_sentinels_does_not_import_pytorch(self):
        script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

assert torch.distributed.GroupMember.WORLD is None
assert torch.distributed.group.WORLD is None
assert torch.distributed.GroupMember.NON_GROUP_MEMBER == -100
assert torch.distributed.get_rank(torch.distributed.GroupMember.NON_GROUP_MEMBER) == -1
assert torch.distributed.get_world_size(torch.distributed.GroupMember.NON_GROUP_MEMBER) == -1
assert torch.distributed.destroy_process_group(
    torch.distributed.GroupMember.NON_GROUP_MEMBER
) is None
assert not hasattr(torch.distributed, "init_process_group")
assert not hasattr(torch.distributed, "all_reduce")
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
