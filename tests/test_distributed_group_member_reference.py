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
class DistributedGroupMemberReferenceTests(unittest.TestCase):
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

    def normalized_class_shape(self, cls):
        try:
            signature = str(inspect.signature(cls)).replace("torch_rs", "torch")
        except ValueError:
            signature = None
        return (
            cls.__module__.replace("torch_rs", "torch"),
            cls.__name__,
            cls.__qualname__,
            cls.__doc__,
            tuple(
                (base.__module__.replace("torch_rs", "torch"), base.__qualname__)
                for base in cls.__bases__
            ),
            tuple(cls.__dict__),
            cls.__annotations__,
            signature,
        )

    def test_classes_metadata_and_values_match_pytorch_2_13(self):
        actual_c10d = torch.distributed.distributed_c10d
        expected_c10d = reference_torch.distributed.distributed_c10d

        for name in ("GroupMember", "group"):
            actual = getattr(actual_c10d, name)
            expected = getattr(expected_c10d, name)
            with self.subTest(name=name):
                self.assertEqual(
                    self.normalized_class_shape(actual),
                    self.normalized_class_shape(expected),
                )
                self.assertEqual(
                    self.normalized_class_shape(type(actual)),
                    self.normalized_class_shape(type(expected)),
                )
                actual_property = type(actual).__dict__["WORLD"]
                expected_property = type(expected).__dict__["WORLD"]
                self.assertIsInstance(actual_property, property)
                self.assertIsInstance(expected_property, property)
                self.assertEqual(actual_property.__doc__, expected_property.__doc__)
                self.assertEqual(
                    str(inspect.signature(actual_property.fget)).replace(
                        "torch_rs", "torch"
                    ),
                    str(inspect.signature(expected_property.fget)),
                )
                self.assertEqual(
                    str(inspect.signature(actual_property.fset)).replace(
                        "torch_rs", "torch"
                    ),
                    str(inspect.signature(expected_property.fset)),
                )
                self.assertIs(actual.WORLD, expected.WORLD)

        self.assertIsNot(actual_c10d.GroupMember, actual_c10d.group)
        self.assertIsNot(expected_c10d.GroupMember, expected_c10d.group)
        self.assertEqual(
            actual_c10d.GroupMember.NON_GROUP_MEMBER,
            expected_c10d.GroupMember.NON_GROUP_MEMBER,
        )
        self.assertIs(
            type(actual_c10d.GroupMember.NON_GROUP_MEMBER),
            type(expected_c10d.GroupMember.NON_GROUP_MEMBER),
        )
        self.assertEqual(
            hasattr(actual_c10d.group, "NON_GROUP_MEMBER"),
            hasattr(expected_c10d.group, "NON_GROUP_MEMBER"),
        )

        for actual, expected in (
            (actual_c10d.GroupMember(), expected_c10d.GroupMember()),
            (actual_c10d.group(), expected_c10d.group()),
        ):
            self.assertEqual(actual.__dict__, expected.__dict__)
            self.assertEqual(hasattr(actual, "WORLD"), hasattr(expected, "WORLD"))
            self.assertEqual(
                hasattr(actual, "NON_GROUP_MEMBER"),
                hasattr(expected, "NON_GROUP_MEMBER"),
            )

    def test_import_copy_and_pickle_behavior_matches(self):
        actual_distributed = torch.distributed
        expected_distributed = reference_torch.distributed
        actual_c10d = actual_distributed.distributed_c10d
        expected_c10d = expected_distributed.distributed_c10d

        for name in ("GroupMember", "group"):
            actual = getattr(actual_c10d, name)
            expected = getattr(expected_c10d, name)
            self.assertIs(getattr(actual_distributed, name), actual)
            self.assertIs(getattr(expected_distributed, name), expected)
            self.assertIs(copy.copy(actual), actual)
            self.assertIs(copy.copy(expected), expected)
            self.assertIs(copy.deepcopy(actual), actual)
            self.assertIs(copy.deepcopy(expected), expected)

            actual_instance = actual()
            expected_instance = expected()
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(name=name, protocol=protocol):
                    self.assertIs(
                        pickle.loads(pickle.dumps(actual, protocol=protocol)),
                        actual,
                    )
                    self.assertIs(
                        pickle.loads(pickle.dumps(expected, protocol=protocol)),
                        expected,
                    )
                    self.assertEqual(
                        self.pickle_shape(actual, protocol),
                        self.pickle_shape(expected, protocol),
                    )
                    actual_restored = pickle.loads(
                        pickle.dumps(actual_instance, protocol=protocol)
                    )
                    expected_restored = pickle.loads(
                        pickle.dumps(expected_instance, protocol=protocol)
                    )
                    self.assertEqual(
                        self.pickle_shape(actual_instance, protocol),
                        self.pickle_shape(expected_instance, protocol),
                    )
                    self.assertIs(type(actual_restored), actual)
                    self.assertIs(type(expected_restored), expected)
                    self.assertEqual(
                        actual_restored.__dict__, expected_restored.__dict__
                    )

        for module, c10d in (
            (actual_distributed, actual_c10d),
            (expected_distributed, expected_c10d),
        ):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertIs(namespace["GroupMember"], c10d.GroupMember)
            self.assertIs(namespace["group"], c10d.group)

            namespace = {}
            exec(f"from {c10d.__name__} import *", namespace)
            self.assertIs(namespace["GroupMember"], c10d.GroupMember)
            self.assertIs(namespace["group"], c10d.group)

    def test_mutated_world_state_matches_pytorch_2_13(self):
        def mutated_state(module, owner_name):
            distributed = module.distributed
            owner = getattr(distributed, owner_name)
            marker = object()
            previous = owner.WORLD
            try:
                owner.WORLD = marker
                return (
                    distributed.GroupMember.WORLD is marker,
                    distributed.group.WORLD is marker,
                    distributed.is_initialized(),
                    distributed.get_group_rank(marker, 7),
                    distributed.get_global_rank(marker, 11),
                )
            finally:
                owner.WORLD = previous

        for owner_name in ("GroupMember", "group"):
            with self.subTest(owner=owner_name):
                self.assertEqual(
                    mutated_state(torch, owner_name),
                    mutated_state(reference_torch, owner_name),
                )
        self.assertIs(torch.distributed.GroupMember.WORLD, None)
        self.assertIs(torch.distributed.group.WORLD, None)
        self.assertIs(torch.distributed.is_initialized(), False)
        self.assertIs(reference_torch.distributed.GroupMember.WORLD, None)
        self.assertIs(reference_torch.distributed.group.WORLD, None)
        self.assertIs(reference_torch.distributed.is_initialized(), False)

    def test_uninitialized_api_outcomes_match(self):
        actual = torch.distributed
        expected = reference_torch.distributed

        for actual_world, expected_world in (
            (actual.GroupMember.WORLD, expected.GroupMember.WORLD),
            (actual.group.WORLD, expected.group.WORLD),
        ):
            for name in (
                "destroy_process_group",
                "get_backend_config",
                "get_backend",
                "get_rank",
                "get_world_size",
            ):
                with self.subTest(name=name, sentinel="WORLD"):
                    self.assertEqual(
                        self.outcome(
                            lambda name=name, value=actual_world: getattr(
                                actual, name
                            )(value)
                        ),
                        self.outcome(
                            lambda name=name, value=expected_world: getattr(
                                expected, name
                            )(value)
                        ),
                    )

        actual_non_member = actual.GroupMember.NON_GROUP_MEMBER
        expected_non_member = expected.GroupMember.NON_GROUP_MEMBER
        for name in (
            "destroy_process_group",
            "get_backend_config",
            "get_backend",
            "get_rank",
            "get_world_size",
        ):
            with self.subTest(name=name, sentinel="NON_GROUP_MEMBER"):
                self.assertEqual(
                    self.outcome(
                        lambda name=name: getattr(actual, name)(actual_non_member)
                    ),
                    self.outcome(
                        lambda name=name: getattr(expected, name)(expected_non_member)
                    ),
                )

        for name in ("get_group_rank", "get_global_rank"):
            self.assertEqual(
                self.outcome(
                    lambda name=name: getattr(actual, name)(actual_non_member, 7)
                ),
                self.outcome(
                    lambda name=name: getattr(expected, name)(expected_non_member, 7)
                ),
            )
        self.assertEqual(
            self.outcome(
                lambda: actual.get_process_group_ranks(actual_non_member)
            ),
            self.outcome(
                lambda: expected.get_process_group_ranks(expected_non_member)
            ),
        )
        self.assertIs(actual.is_initialized(), False)
        self.assertIs(expected.is_initialized(), False)
        self.assertEqual(actual.get_pg_count(), expected.get_pg_count())


if __name__ == "__main__":
    unittest.main()
