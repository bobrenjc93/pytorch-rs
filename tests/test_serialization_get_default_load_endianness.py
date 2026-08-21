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


class _RuntimeLoadEndianness(enum.Enum):
    NATIVE = 1
    LITTLE = 2
    BIG = 3


class SerializationDefaultLoadEndiannessTests(unittest.TestCase):
    def test_enum_has_exact_members_values_and_identity(self):
        endianness = torch.serialization.LoadEndianness
        expected = (
            ("NATIVE", 1),
            ("LITTLE", 2),
            ("BIG", 3),
        )

        self.assertIs(type(endianness), type(enum.Enum))
        self.assertEqual(endianness.__bases__, (enum.Enum,))
        self.assertEqual(endianness.__name__, "LoadEndianness")
        self.assertEqual(endianness.__qualname__, "LoadEndianness")
        self.assertEqual(endianness.__module__, "torch_rs.serialization")
        self.assertIs(inspect.getmodule(endianness), torch.serialization)
        self.assertEqual(endianness.__doc__, _RuntimeLoadEndianness.__doc__)
        self.assertEqual(endianness.__annotations__, {})
        self.assertEqual(
            tuple(endianness.__members__),
            tuple(name for name, _ in expected),
        )
        self.assertEqual(
            tuple((member.name, member.value) for member in endianness),
            expected,
        )
        self.assertEqual(len(endianness), 3)

        for name, value in expected:
            with self.subTest(name=name):
                member = getattr(endianness, name)
                self.assertIs(endianness.__members__[name], member)
                self.assertIs(endianness[name], member)
                self.assertIs(endianness(value), member)
                self.assertIs(type(member.value), int)
                self.assertIs(type(member), endianness)
                self.assertFalse(isinstance(member, int))
                self.assertEqual(str(member), f"LoadEndianness.{name}")
                self.assertEqual(repr(member), f"<LoadEndianness.{name}: {value}>")

    def test_default_is_exact_none_and_preserves_grad_mode(self):
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

    def test_default_and_enum_singletons_are_stable_across_threads(self):
        serialization = torch.serialization
        endianness = serialization.LoadEndianness
        members = tuple(endianness)
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
                        serialization.get_default_load_endianness() is None,
                        all(
                            getattr(endianness, member.name) is member
                            for member in members
                        ),
                        serialization.get_default_load_endianness() is None,
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
                    expected_grad_state,
                ),
            )

    def test_signature_annotations_documentation_and_module_identity(self):
        serialization = importlib.import_module("torch_rs.serialization")
        endianness = serialization.LoadEndianness
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
        self.assertEqual(function.__annotations__, {"return": endianness | None})
        self.assertEqual(
            typing.get_type_hints(function),
            {"return": endianness | None},
        )
        self.assertEqual(function.__name__, "get_default_load_endianness")
        self.assertEqual(function.__qualname__, "get_default_load_endianness")
        self.assertEqual(
            inspect.cleandoc(function.__doc__),
            inspect.cleandoc(GETTER_DOC),
        )

    def test_imports_exports_copy_and_pickle_use_the_canonical_module(self):
        serialization = torch.serialization
        endianness = serialization.LoadEndianness
        function = serialization.get_default_load_endianness
        exported_names = [
            "LoadEndianness",
            "get_crc32_options",
            "set_crc32_options",
            "get_default_load_endianness",
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
        self.assertIs(direct_import["LoadEndianness"], endianness)
        self.assertIs(direct_import["get_default_load_endianness"], function)

        serialization_namespace = {}
        exec("from torch_rs.serialization import *", serialization_namespace)
        self.assertEqual(
            {
                name
                for name in serialization_namespace
                if not name.startswith("__")
            },
            set(exported_names),
        )
        self.assertIs(serialization_namespace["LoadEndianness"], endianness)
        self.assertIs(
            serialization_namespace["get_default_load_endianness"],
            function,
        )

        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        for name in ("serialization", "LoadEndianness", function.__name__):
            with self.subTest(top_level_name=name):
                self.assertNotIn(name, torch.__all__)
                self.assertNotIn(name, top_level_namespace)
        self.assertFalse(hasattr(torch, "LoadEndianness"))
        self.assertFalse(hasattr(torch, "get_default_load_endianness"))

        objects = (endianness, function, *endianness)
        for value in objects:
            with self.subTest(value=value):
                self.assertIs(copy.copy(value), value)
                self.assertIs(copy.deepcopy(value), value)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            for value in objects:
                with self.subTest(protocol=protocol, value=value):
                    payload = pickle.dumps(value, protocol=protocol)
                    self.assertIn(b"torch_rs.serialization", payload)
                    self.assertIs(pickle.loads(payload), value)

    def test_argument_errors_match_pytorch_2_13(self):
        function = torch.serialization.get_default_load_endianness
        cases = (
            (
                lambda: function(None),
                "get_default_load_endianness() takes 0 positional arguments "
                "but 1 was given",
            ),
            (
                lambda: function(None, None),
                "get_default_load_endianness() takes 0 positional arguments "
                "but 2 were given",
            ),
            (
                lambda: function(endianness=None),
                "get_default_load_endianness() got an unexpected keyword "
                "argument 'endianness'",
            ),
            (
                lambda: function(None, endianness=None),
                "get_default_load_endianness() got an unexpected keyword "
                "argument 'endianness'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertIsNone(function())

        self.assertIsNone(function(**{}))

    def test_enum_rejects_invalid_construction_without_changing_the_default(self):
        endianness = torch.serialization.LoadEndianness
        with self.assertRaises((TypeError, ValueError)) as runtime_raised:
            _RuntimeLoadEndianness(1, 2)

        cases = (
            (TypeError, lambda: endianness()),
            (type(runtime_raised.exception), lambda: endianness(1, 2)),
            (TypeError, lambda: endianness(name="NATIVE")),
            (ValueError, lambda: endianness(0)),
            (ValueError, lambda: endianness("NATIVE")),
            (KeyError, lambda: endianness["native"]),
        )

        self.assertIs(endianness(value=1), endianness.NATIVE)
        for error_type, call in cases:
            with self.subTest(error_type=error_type.__name__):
                with self.assertRaises(error_type):
                    call()
                self.assertIsNone(
                    torch.serialization.get_default_load_endianness()
                )

    def test_reload_and_reimport_recreate_the_enum_and_preserve_the_default(self):
        original_module = torch.serialization
        original_class = original_module.LoadEndianness
        original_function = original_module.get_default_load_endianness
        original_members = tuple(original_class)
        module_name = original_module.__name__

        self.assertIs(importlib.reload(original_module), original_module)
        self.assertIs(torch.serialization, original_module)
        self.assertIsNot(original_module.LoadEndianness, original_class)
        self.assertIsNot(
            original_module.get_default_load_endianness,
            original_function,
        )
        self.assertEqual(
            tuple((member.name, member.value) for member in original_class),
            tuple(
                (member.name, member.value)
                for member in original_module.LoadEndianness
            ),
        )
        self.assertEqual(tuple(original_class), original_members)
        self.assertIsNone(original_function())
        self.assertIsNone(original_module.get_default_load_endianness())

        try:
            self.assertIs(sys.modules.pop(module_name), original_module)
            replacement_module = importlib.import_module(module_name)

            self.assertIsNot(replacement_module, original_module)
            self.assertIs(sys.modules[module_name], replacement_module)
            self.assertIs(torch.serialization, replacement_module)
            self.assertIsNot(
                replacement_module.LoadEndianness,
                original_module.LoadEndianness,
            )
            self.assertIsNone(original_function())
            self.assertIsNone(
                original_module.get_default_load_endianness()
            )
            self.assertIsNone(
                replacement_module.get_default_load_endianness()
            )
        finally:
            sys.modules[module_name] = original_module
            torch.serialization = original_module

    def test_setter_save_and_load_remain_unsupported(self):
        serialization = torch.serialization

        self.assertTrue(hasattr(serialization, "LoadEndianness"))
        self.assertTrue(hasattr(serialization, "get_default_load_endianness"))
        for name in ("set_default_load_endianness", "save", "load"):
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
endianness = serialization.LoadEndianness
assert tuple(endianness.__members__) == ("NATIVE", "LITTLE", "BIG")
assert tuple(member.value for member in endianness) == (1, 2, 3)
assert serialization.get_default_load_endianness() is None
assert copy.copy(endianness.LITTLE) is endianness.LITTLE
assert pickle.loads(pickle.dumps(endianness.BIG)) is endianness.BIG
assert importlib.import_module("torch_rs.serialization") is serialization
assert importlib.reload(serialization) is serialization
assert serialization.LoadEndianness is not endianness
assert serialization.get_default_load_endianness() is None
assert not hasattr(serialization, "set_default_load_endianness")
assert not hasattr(serialization, "save")
assert not hasattr(serialization, "load")
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
