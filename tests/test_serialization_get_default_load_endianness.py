import contextlib
import copy
import enum
import importlib
import inspect
import pickle
import subprocess
import sys
import threading
import types
import typing
import unittest

import torch_rs as torch


GETTER_DOC = """
    Get fallback byte order for loading files

    If byteorder mark is not present in saved checkpoint,
    this byte order is used as fallback.
    By default, it's "native" byte order.

    Returns:
        default_load_endian: Optional[LoadEndianness]
    """


class SerializationDefaultLoadEndiannessTests(unittest.TestCase):
    def test_enum_identity_values_repr_iteration_order_and_aliases(self):
        load_endianness = torch.serialization.LoadEndianness
        expected = (
            ("NATIVE", 1),
            ("LITTLE", 2),
            ("BIG", 3),
        )
        reference_enum = enum.Enum("LoadEndianness", dict(expected))
        members = tuple(load_endianness)

        self.assertIs(type(load_endianness), enum.EnumMeta)
        self.assertEqual(load_endianness.__bases__, (enum.Enum,))
        self.assertEqual(load_endianness.__name__, "LoadEndianness")
        self.assertEqual(load_endianness.__qualname__, "LoadEndianness")
        self.assertEqual(load_endianness.__module__, "torch_rs.serialization")
        self.assertIs(inspect.getmodule(load_endianness), torch.serialization)
        self.assertEqual(load_endianness.__doc__, reference_enum.__doc__)
        self.assertEqual(load_endianness.__annotations__, {})

        self.assertEqual(len(load_endianness), 3)
        self.assertEqual(tuple(reversed(load_endianness)), tuple(reversed(members)))
        self.assertEqual(
            tuple(load_endianness.__members__),
            tuple(name for name, _ in expected),
        )
        self.assertEqual(tuple(load_endianness.__members__.values()), members)
        self.assertEqual(len(set(load_endianness.__members__.values())), 3)

        for member, (name, value) in zip(members, expected):
            with self.subTest(name=name):
                self.assertIs(type(member), load_endianness)
                self.assertIs(getattr(load_endianness, name), member)
                self.assertIs(load_endianness.__members__[name], member)
                self.assertIs(load_endianness[name], member)
                self.assertIs(load_endianness(value), member)
                self.assertEqual(member.name, name)
                self.assertIs(type(member.value), int)
                self.assertEqual(member.value, value)
                self.assertEqual(str(member), f"LoadEndianness.{name}")
                self.assertEqual(repr(member), f"<LoadEndianness.{name}: {value}>")
                self.assertTrue(member)
                self.assertNotEqual(member, value)

    def test_default_none_is_exact_and_preserves_grad_mode(self):
        function = torch.serialization.get_default_load_endianness
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

        def assert_query_preserves_grad_mode(expected_grad_state):
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)
            self.assertIsNone(function())
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)

        assert_query_preserves_grad_mode(True)
        with torch.no_grad():
            assert_query_preserves_grad_mode(False)
            with torch.no_grad():
                assert_query_preserves_grad_mode(False)
            assert_query_preserves_grad_mode(False)
        assert_query_preserves_grad_mode(True)

    def test_default_and_enum_members_are_stable_across_threads(self):
        function = torch.serialization.get_default_load_endianness
        load_endianness = torch.serialization.LoadEndianness
        expected_members = tuple(load_endianness)
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = torch.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    results[index] = (
                        torch.is_grad_enabled(),
                        function() is None,
                        tuple(load_endianness) == expected_members,
                        load_endianness.NATIVE is expected_members[0],
                        load_endianness.LITTLE is expected_members[1],
                        load_endianness.BIG is expected_members[2],
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
            expected_grad_state = index % 2 == 0
            self.assertEqual(
                result,
                (
                    expected_grad_state,
                    True,
                    True,
                    True,
                    True,
                    True,
                    expected_grad_state,
                ),
            )

    def test_signature_annotations_documentation_and_module_identity(self):
        serialization = importlib.import_module("torch_rs.serialization")
        load_endianness = serialization.LoadEndianness
        function = serialization.get_default_load_endianness

        self.assertIs(torch.serialization, serialization)
        self.assertIs(sys.modules["torch_rs.serialization"], serialization)
        self.assertIsNone(serialization.__doc__)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__module__, "torch_rs.serialization")
        self.assertIs(inspect.getmodule(function), serialization)
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(
            str(inspect.signature(function)),
            "() -> torch_rs.serialization.LoadEndianness | None",
        )
        self.assertEqual(
            function.__annotations__,
            {"return": load_endianness | None},
        )
        self.assertEqual(
            typing.get_type_hints(function),
            {"return": load_endianness | None},
        )
        self.assertEqual(function.__name__, "get_default_load_endianness")
        self.assertEqual(function.__qualname__, "get_default_load_endianness")
        self.assertEqual(
            inspect.cleandoc(function.__doc__),
            inspect.cleandoc(GETTER_DOC),
        )

    def test_imports_exports_copy_and_pickle_use_canonical_objects(self):
        serialization = torch.serialization
        load_endianness = serialization.LoadEndianness
        function = serialization.get_default_load_endianness
        exported_names = [
            "LoadEndianness",
            "get_crc32_options",
            "set_crc32_options",
            "get_default_load_endianness",
            "set_default_load_endianness",
            "get_default_mmap_options",
            "set_default_mmap_options",
        ]

        self.assertEqual(serialization.__all__, exported_names)

        package_import = {}
        exec("from torch_rs import serialization", package_import)
        self.assertIs(package_import["serialization"], serialization)

        direct_import = {}
        exec(
            "from torch_rs.serialization import "
            "LoadEndianness, get_default_load_endianness",
            direct_import,
        )
        self.assertIs(direct_import["LoadEndianness"], load_endianness)
        self.assertIs(direct_import["get_default_load_endianness"], function)

        namespace = {}
        exec("from torch_rs.serialization import *", namespace)
        self.assertEqual(
            {name for name in namespace if not name.startswith("__")},
            set(exported_names),
        )
        self.assertIs(namespace["LoadEndianness"], load_endianness)
        self.assertIs(namespace["get_default_load_endianness"], function)

        self.assertNotIn("serialization", torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("serialization", top_level_namespace)
        for name in ("LoadEndianness", function.__name__):
            self.assertFalse(hasattr(torch, name))
            self.assertNotIn(name, torch.__all__)
            self.assertNotIn(name, top_level_namespace)

        objects = (load_endianness, function, *tuple(load_endianness))
        for value in objects:
            self.assertIs(copy.copy(value), value)
            self.assertIs(copy.deepcopy(value), value)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(value=repr(value), protocol=protocol):
                    payload = pickle.dumps(value, protocol=protocol)
                    self.assertIn(b"torch_rs.serialization", payload)
                    self.assertIs(pickle.loads(payload), value)

    def test_getter_and_enum_argument_errors_match_pytorch_2_13(self):
        function = torch.serialization.get_default_load_endianness
        load_endianness = torch.serialization.LoadEndianness
        getter_cases = (
            (
                lambda: function(None),
                "get_default_load_endianness() takes 0 positional arguments but 1 "
                "was given",
            ),
            (
                lambda: function(None, None),
                "get_default_load_endianness() takes 0 positional arguments but 2 "
                "were given",
            ),
            (
                lambda: function(enabled=True),
                "get_default_load_endianness() got an unexpected keyword argument "
                "'enabled'",
            ),
            (
                lambda: function(None, enabled=True),
                "get_default_load_endianness() got an unexpected keyword argument "
                "'enabled'",
            ),
        )
        for call, message in getter_cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertIsNone(function())

        invalid_values = (
            (0, "0 is not a valid LoadEndianness"),
            (4, "4 is not a valid LoadEndianness"),
            (None, "None is not a valid LoadEndianness"),
            ("NATIVE", "'NATIVE' is not a valid LoadEndianness"),
        )
        for value, message in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError) as raised:
                    load_endianness(value)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

        reference_enum = enum.Enum(
            "LoadEndianness",
            {"NATIVE": 1, "LITTLE": 2, "BIG": 3},
        )
        with self.assertRaises(Exception) as raised:
            load_endianness(1, 2)
        with self.assertRaises(Exception) as expected_raised:
            reference_enum(1, 2)
        self.assertIs(type(raised.exception), type(expected_raised.exception))
        self.assertEqual(str(raised.exception), str(expected_raised.exception))
        self.assertEqual(raised.exception.args, expected_raised.exception.args)

        with self.assertRaises(KeyError) as raised:
            load_endianness["native"]
        self.assertEqual(raised.exception.args, ("native",))
        self.assertIs(load_endianness(value=1), load_endianness.NATIVE)
        self.assertIsNone(function(**{}))

    def test_reload_and_reimport_rebind_the_enum_and_keep_the_default(self):
        original_module = torch.serialization
        original_class = original_module.LoadEndianness
        original_members = tuple(original_class)
        original_getter = original_module.get_default_load_endianness
        module_name = original_module.__name__

        self.assertIs(importlib.reload(original_module), original_module)
        reloaded_class = original_module.LoadEndianness
        reloaded_getter = original_module.get_default_load_endianness
        self.assertIs(torch.serialization, original_module)
        self.assertIsNot(reloaded_class, original_class)
        self.assertIsNot(reloaded_getter, original_getter)
        self.assertEqual([member.value for member in reloaded_class], [1, 2, 3])
        self.assertTrue(
            all(
                old is not new
                for old, new in zip(original_members, tuple(reloaded_class))
            )
        )
        self.assertIsNone(original_getter())
        self.assertIsNone(reloaded_getter())

        try:
            self.assertIs(sys.modules.pop(module_name), original_module)
            replacement_module = importlib.import_module(module_name)

            self.assertIsNot(replacement_module, original_module)
            self.assertIs(sys.modules[module_name], replacement_module)
            self.assertIs(torch.serialization, replacement_module)
            self.assertIsNot(
                replacement_module.LoadEndianness,
                reloaded_class,
            )
            self.assertEqual(
                [member.value for member in replacement_module.LoadEndianness],
                [1, 2, 3],
            )
            self.assertIsNone(original_getter())
            self.assertIsNone(reloaded_getter())
            self.assertIsNone(
                replacement_module.get_default_load_endianness()
            )
        finally:
            sys.modules[module_name] = original_module
            torch.serialization = original_module

    def test_save_and_load_remain_unsupported(self):
        serialization = torch.serialization

        self.assertEqual(
            {name for name in vars(serialization) if not name.startswith("_")},
            {
                "LoadEndianness",
                "get_crc32_options",
                "set_crc32_options",
                "get_default_load_endianness",
                "set_default_load_endianness",
                "get_default_mmap_options",
                "set_default_mmap_options",
            },
        )
        for name in ("save", "load"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(serialization, name))
                self.assertNotIn(name, serialization.__all__)

    def test_importing_reloading_and_calling_does_not_import_pytorch(self):
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

serialization = torch.serialization
load_endianness = serialization.LoadEndianness
getter = serialization.get_default_load_endianness
members = tuple(load_endianness)
assert [member.name for member in members] == ["NATIVE", "LITTLE", "BIG"]
assert [member.value for member in members] == [1, 2, 3]
assert getter() is None
for value in (load_endianness, getter, *members):
    assert copy.copy(value) is value
    assert copy.deepcopy(value) is value
    assert pickle.loads(pickle.dumps(value)) is value

assert importlib.reload(serialization) is serialization
assert serialization.LoadEndianness is not load_endianness
assert serialization.get_default_load_endianness() is None
old_reloaded_getter = serialization.get_default_load_endianness
del sys.modules["torch_rs.serialization"]
replacement = importlib.import_module("torch_rs.serialization")
assert replacement is not serialization
assert torch.serialization is replacement
assert replacement.LoadEndianness is not serialization.LoadEndianness
assert getter() is None
assert old_reloaded_getter() is None
assert replacement.get_default_load_endianness() is None
assert hasattr(replacement, "set_default_load_endianness")
assert not hasattr(replacement, "save")
assert not hasattr(replacement, "load")
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
