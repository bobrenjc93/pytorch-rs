import contextlib
import copy
import enum
import importlib
import inspect
import pickle
import pickletools
import sys
import threading
import types
import typing
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class SerializationDefaultLoadEndiannessReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "serialization load-endianness differentials require pinned "
                "PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def enum_outcome(self, module):
        load_endianness = module.serialization.LoadEndianness
        members = tuple(load_endianness)
        names = tuple(load_endianness.__members__)
        return (
            type(load_endianness).__module__,
            type(load_endianness).__qualname__,
            tuple(
                (base.__module__, base.__qualname__)
                for base in load_endianness.__bases__
            ),
            load_endianness.__name__,
            load_endianness.__qualname__,
            load_endianness.__doc__,
            load_endianness.__annotations__,
            names,
            tuple(member.name for member in members),
            tuple(member.value for member in members),
            tuple(type(member.value).__name__ for member in members),
            tuple(str(member) for member in members),
            tuple(repr(member) for member in members),
            tuple(
                load_endianness[name] is member
                for name, member in zip(names, members)
            ),
            tuple(
                load_endianness(member.value) is member for member in members
            ),
            tuple(
                load_endianness.__members__[name] is member
                for name, member in zip(names, members)
            ),
            len(set(load_endianness.__members__.values())),
            tuple(reversed(load_endianness)) == tuple(reversed(members)),
        )

    def threaded_default_outcome(self, module):
        function = module.serialization.get_default_load_endianness
        load_endianness = module.serialization.LoadEndianness
        members = tuple(load_endianness)

        def query_outcome():
            before = module.is_grad_enabled()
            result = function()
            after = module.is_grad_enabled()
            return (
                before,
                result,
                result is None,
                load_endianness.NATIVE is members[0],
                load_endianness.LITTLE is members[1],
                load_endianness.BIG is members[2],
                after,
            )

        states = [query_outcome()]
        with module.no_grad():
            states.append(query_outcome())
            with module.no_grad():
                states.append(query_outcome())
            states.append(query_outcome())
        states.append(query_outcome())

        worker_count = 8
        barrier = threading.Barrier(worker_count)
        worker_states = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = module.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    worker_states[index] = query_outcome()
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))

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
        return states, worker_states

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

    def reload_outcome(self, module):
        original_module = module.serialization
        original_class = original_module.LoadEndianness
        original_members = tuple(original_class)
        original_getter = original_module.get_default_load_endianness
        module_name = original_module.__name__

        reloaded = importlib.reload(original_module)
        reloaded_class = reloaded.LoadEndianness
        reloaded_getter = reloaded.get_default_load_endianness
        after_reload = (
            reloaded is original_module,
            module.serialization is original_module,
            reloaded_class is not original_class,
            reloaded_getter is not original_getter,
            tuple(member.value for member in reloaded_class),
            tuple(
                old is not new
                for old, new in zip(original_members, tuple(reloaded_class))
            ),
            original_getter(),
            reloaded_getter(),
        )

        try:
            removed = sys.modules.pop(module_name)
            replacement_module = importlib.import_module(module_name)
            after_reimport = (
                removed is original_module,
                replacement_module is not original_module,
                sys.modules[module_name] is replacement_module,
                module.serialization is replacement_module,
                replacement_module.LoadEndianness is not reloaded_class,
                tuple(
                    member.value
                    for member in replacement_module.LoadEndianness
                ),
                original_getter(),
                reloaded_getter(),
                replacement_module.get_default_load_endianness(),
            )
        finally:
            sys.modules[module_name] = original_module
            module.serialization = original_module

        return after_reload, after_reimport

    def test_enum_identity_values_repr_iteration_and_aliases_match(self):
        actual = self.enum_outcome(torch)
        expected = self.enum_outcome(reference_torch)

        self.assertEqual(actual, expected)
        self.assertEqual(actual[7], ("NATIVE", "LITTLE", "BIG"))
        self.assertEqual(actual[9], (1, 2, 3))
        self.assertEqual(actual[16], 3)
        self.assertIs(
            type(torch.serialization.LoadEndianness),
            enum.EnumType,
        )

    def test_default_threaded_and_grad_states_match_pytorch_2_13(self):
        self.assertEqual(
            self.threaded_default_outcome(torch),
            self.threaded_default_outcome(reference_torch),
        )
        self.assertIsNone(torch.serialization.get_default_load_endianness())
        self.assertIsNone(
            reference_torch.serialization.get_default_load_endianness()
        )

    def test_signature_annotations_documentation_and_identity_match(self):
        actual_module = importlib.import_module("torch_rs.serialization")
        expected_module = importlib.import_module("torch.serialization")
        actual_class = actual_module.LoadEndianness
        expected_class = expected_module.LoadEndianness
        actual = actual_module.get_default_load_endianness
        expected = expected_module.get_default_load_endianness

        self.assertIs(torch.serialization, actual_module)
        self.assertIs(reference_torch.serialization, expected_module)
        self.assertIsNone(actual_module.__doc__)
        self.assertEqual(actual_module.__doc__, expected_module.__doc__)
        self.assertIs(type(actual_class), type(expected_class))
        self.assertEqual(actual_class.__bases__, expected_class.__bases__)
        self.assertEqual(actual_class.__name__, expected_class.__name__)
        self.assertEqual(actual_class.__qualname__, expected_class.__qualname__)
        self.assertEqual(
            actual_class.__module__.replace("torch_rs", "torch"),
            expected_class.__module__,
        )
        self.assertIs(inspect.getmodule(actual_class), actual_module)
        self.assertIs(inspect.getmodule(expected_class), expected_module)
        self.assertEqual(actual_class.__doc__, expected_class.__doc__)
        self.assertEqual(actual_class.__annotations__, expected_class.__annotations__)
        self.assertEqual(
            str(inspect.signature(actual_class)),
            str(inspect.signature(expected_class)),
        )

        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual)).replace("torch_rs", "torch"),
            str(inspect.signature(expected)),
        )
        self.assertEqual(
            actual.__annotations__,
            {"return": actual_class | None},
        )
        self.assertEqual(
            expected.__annotations__,
            {"return": expected_class | None},
        )
        self.assertEqual(
            typing.get_type_hints(actual),
            {"return": actual_class | None},
        )
        self.assertEqual(
            typing.get_type_hints(expected),
            {"return": expected_class | None},
        )
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"),
            expected.__module__,
        )
        self.assertIs(inspect.getmodule(actual), actual_module)
        self.assertIs(inspect.getmodule(expected), expected_module)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )

    def test_imports_exports_copy_and_pickle_match_pytorch_2_13(self):
        actual_module = torch.serialization
        expected_module = reference_torch.serialization
        supported_names = (
            "LoadEndianness",
            "get_crc32_options",
            "set_crc32_options",
            "get_default_load_endianness",
            "get_default_mmap_options",
            "set_default_mmap_options",
        )

        self.assertEqual(
            actual_module.__all__,
            [name for name in expected_module.__all__ if name in supported_names],
        )

        actual_package_import = {}
        expected_package_import = {}
        exec("from torch_rs import serialization", actual_package_import)
        exec("from torch import serialization", expected_package_import)
        self.assertIs(actual_package_import["serialization"], actual_module)
        self.assertIs(expected_package_import["serialization"], expected_module)

        actual_direct_import = {}
        expected_direct_import = {}
        exec(
            "from torch_rs.serialization import "
            "LoadEndianness, get_default_load_endianness",
            actual_direct_import,
        )
        exec(
            "from torch.serialization import "
            "LoadEndianness, get_default_load_endianness",
            expected_direct_import,
        )
        self.assertIs(
            actual_direct_import["LoadEndianness"],
            actual_module.LoadEndianness,
        )
        self.assertIs(
            expected_direct_import["LoadEndianness"],
            expected_module.LoadEndianness,
        )
        self.assertIs(
            actual_direct_import["get_default_load_endianness"],
            actual_module.get_default_load_endianness,
        )
        self.assertIs(
            expected_direct_import["get_default_load_endianness"],
            expected_module.get_default_load_endianness,
        )

        for module in (actual_module, expected_module):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertIs(namespace["LoadEndianness"], module.LoadEndianness)
            self.assertIs(
                namespace["get_default_load_endianness"],
                module.get_default_load_endianness,
            )

        object_pairs = (
            (actual_module.LoadEndianness, expected_module.LoadEndianness),
            (
                actual_module.get_default_load_endianness,
                expected_module.get_default_load_endianness,
            ),
            *zip(actual_module.LoadEndianness, expected_module.LoadEndianness),
        )
        for actual, expected in object_pairs:
            self.assertIs(copy.copy(actual), actual)
            self.assertIs(copy.copy(expected), expected)
            self.assertIs(copy.deepcopy(actual), actual)
            self.assertIs(copy.deepcopy(expected), expected)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(value=repr(actual), protocol=protocol):
                    self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                    self.assertIs(
                        pickle.loads(pickle.dumps(expected, protocol)),
                        expected,
                    )
                    self.assertEqual(
                        self.pickle_shape(actual, protocol),
                        self.pickle_shape(expected, protocol),
                    )

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("LoadEndianness", namespace)
            self.assertNotIn("get_default_load_endianness", namespace)

    def test_getter_and_enum_argument_errors_match_pytorch_2_13(self):
        actual_getter = torch.serialization.get_default_load_endianness
        expected_getter = (
            reference_torch.serialization.get_default_load_endianness
        )
        actual_enum = torch.serialization.LoadEndianness
        expected_enum = reference_torch.serialization.LoadEndianness
        cases = (
            (lambda: actual_getter(None), lambda: expected_getter(None)),
            (
                lambda: actual_getter(None, None),
                lambda: expected_getter(None, None),
            ),
            (
                lambda: actual_getter(enabled=True),
                lambda: expected_getter(enabled=True),
            ),
            (
                lambda: actual_getter(None, enabled=True),
                lambda: expected_getter(None, enabled=True),
            ),
            (lambda: actual_enum(), lambda: expected_enum()),
            (lambda: actual_enum(0), lambda: expected_enum(0)),
            (lambda: actual_enum(4), lambda: expected_enum(4)),
            (lambda: actual_enum(None), lambda: expected_enum(None)),
            (lambda: actual_enum("NATIVE"), lambda: expected_enum("NATIVE")),
            (lambda: actual_enum(1, 2), lambda: expected_enum(1, 2)),
            (lambda: actual_enum(foo=1), lambda: expected_enum(foo=1)),
            (lambda: actual_enum["native"], lambda: expected_enum["native"]),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)
                self.assertIsNone(actual_getter())
                self.assertIsNone(expected_getter())

        for name, value in (("NATIVE", 1), ("LITTLE", 2), ("BIG", 3)):
            self.assertIs(actual_enum(value), actual_enum[name])
            self.assertIs(expected_enum(value), expected_enum[name])
        self.assertIs(actual_enum(value=1), actual_enum.NATIVE)
        self.assertIs(expected_enum(value=1), expected_enum.NATIVE)
        self.assertIsNone(actual_getter(**{}))
        self.assertIsNone(expected_getter(**{}))

    def test_reference_only_setter_bounds_unsupported_nondefault_states(self):
        actual_module = torch.serialization
        expected_module = reference_torch.serialization
        actual_getter = actual_module.get_default_load_endianness
        expected_getter = expected_module.get_default_load_endianness
        setter = expected_module.set_default_load_endianness
        original = expected_getter()

        self.assertFalse(hasattr(actual_module, "set_default_load_endianness"))
        self.assertIsNone(actual_getter())
        try:
            self.assertIsNone(setter(None))
            self.assertIsNone(expected_getter())
            for member in expected_module.LoadEndianness:
                with self.subTest(member=member.name):
                    self.assertIsNone(setter(member))
                    self.assertIs(expected_getter(), member)
                    self.assertIsNone(actual_getter())

            for value in (1, 2, 3, "native", "little", "big", True, object()):
                with self.subTest(invalid=repr(value)):
                    before = expected_getter()
                    with self.assertRaises(TypeError) as raised:
                        setter(value)
                    message = (
                        "Invalid argument type in function "
                        "set_default_load_endianness"
                    )
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))
                    self.assertIs(expected_getter(), before)
                    self.assertIsNone(actual_getter())
        finally:
            setter(original)

    def test_reload_and_reimport_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_outcome(torch),
            self.reload_outcome(reference_torch),
        )

    def test_setter_save_and_load_remain_unsupported(self):
        actual_module = torch.serialization
        expected_module = reference_torch.serialization
        actual_public = {
            name for name in vars(actual_module) if not name.startswith("_")
        }

        self.assertEqual(
            actual_public,
            {
                "LoadEndianness",
                "get_crc32_options",
                "set_crc32_options",
                "get_default_load_endianness",
                "get_default_mmap_options",
                "set_default_mmap_options",
            },
        )
        self.assertTrue(hasattr(expected_module, "set_default_load_endianness"))
        for name in ("set_default_load_endianness", "save", "load"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(actual_module, name))
                self.assertNotIn(name, actual_module.__all__)

        for name in ("LoadEndianness", "get_default_load_endianness"):
            with self.subTest(top_level_name=name):
                self.assertFalse(hasattr(torch, name))
                self.assertFalse(hasattr(reference_torch, name))
                self.assertNotIn(name, torch.__all__)


if __name__ == "__main__":
    unittest.main()
