import contextlib
import copy
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
                "serialization load endianness differentials require pinned "
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

    def supported_state_outcome(self, module):
        serialization = module.serialization
        endianness = serialization.LoadEndianness
        function = serialization.get_default_load_endianness
        canonical_members = tuple(endianness)

        def query_outcome():
            before = module.is_grad_enabled()
            result = function()
            after = module.is_grad_enabled()
            return (
                before,
                result is None,
                type(result).__module__,
                type(result).__qualname__,
                tuple(
                    (
                        member.name,
                        member.value,
                        endianness(member.value) is member,
                        endianness[member.name] is member,
                    )
                    for member in canonical_members
                ),
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

    def reload_outcome(self, module):
        original_module = module.serialization
        original_class = original_module.LoadEndianness
        original_function = original_module.get_default_load_endianness
        original_members = tuple(original_class)
        module_name = original_module.__name__

        reloaded = importlib.reload(original_module)
        after_reload = (
            reloaded is original_module,
            module.serialization is original_module,
            original_module.LoadEndianness is not original_class,
            original_module.get_default_load_endianness is not original_function,
            original_function() is None,
            original_module.get_default_load_endianness() is None,
            tuple((member.name, member.value) for member in original_class),
            tuple(
                (member.name, member.value)
                for member in original_module.LoadEndianness
            ),
            tuple(original_class) == original_members,
        )

        try:
            removed = sys.modules.pop(module_name)
            replacement_module = importlib.import_module(module_name)
            after_reimport = (
                removed is original_module,
                replacement_module is not original_module,
                sys.modules[module_name] is replacement_module,
                module.serialization is replacement_module,
                replacement_module.LoadEndianness
                is not original_module.LoadEndianness,
                original_function() is None,
                original_module.get_default_load_endianness() is None,
                replacement_module.get_default_load_endianness() is None,
                tuple(
                    (member.name, member.value)
                    for member in replacement_module.LoadEndianness
                ),
            )
        finally:
            sys.modules[module_name] = original_module
            module.serialization = original_module

        return after_reload, after_reimport

    def test_supported_default_threaded_and_grad_states_match_pytorch_2_13(self):
        getter = reference_torch.serialization.get_default_load_endianness
        setter = reference_torch.serialization.set_default_load_endianness
        original = getter()
        try:
            setter(None)
            self.assertEqual(
                self.supported_state_outcome(torch),
                self.supported_state_outcome(reference_torch),
            )
        finally:
            setter(original)

        self.assertIsNone(torch.serialization.get_default_load_endianness())

    def test_reference_only_setter_bounds_unsupported_non_default_states(self):
        actual_getter = torch.serialization.get_default_load_endianness
        expected_serialization = reference_torch.serialization
        expected_getter = expected_serialization.get_default_load_endianness
        expected_setter = expected_serialization.set_default_load_endianness
        original = expected_getter()

        try:
            actual_states = []
            expected_states = []
            requested_states = (
                None,
                expected_serialization.LoadEndianness.NATIVE,
                expected_serialization.LoadEndianness.LITTLE,
                expected_serialization.LoadEndianness.BIG,
                None,
            )
            for state in requested_states:
                self.assertIsNone(expected_setter(state))
                actual_states.append(actual_getter())
                expected_states.append(expected_getter())
        finally:
            expected_setter(original)

        for state in actual_states:
            self.assertIsNone(state)
        self.assertIsNone(expected_states[0])
        self.assertIs(expected_states[1], requested_states[1])
        self.assertIs(expected_states[2], requested_states[2])
        self.assertIs(expected_states[3], requested_states[3])
        self.assertIsNone(expected_states[4])

    def test_enum_shape_and_getter_metadata_match_pytorch_2_13(self):
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
        self.assertEqual(
            tuple(actual_class.__members__),
            tuple(expected_class.__members__),
        )

        for actual_member, expected_member in zip(actual_class, expected_class):
            with self.subTest(member=actual_member.name):
                self.assertEqual(actual_member.name, expected_member.name)
                self.assertEqual(actual_member.value, expected_member.value)
                self.assertIs(type(actual_member.value), type(expected_member.value))
                self.assertEqual(str(actual_member), str(expected_member))
                self.assertEqual(repr(actual_member), repr(expected_member))
                self.assertFalse(isinstance(actual_member, int))
                self.assertFalse(isinstance(expected_member, int))
                self.assertIs(actual_class(actual_member.value), actual_member)
                self.assertIs(expected_class(expected_member.value), expected_member)

        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual)).replace("torch_rs", "torch"),
            str(inspect.signature(expected)),
        )
        self.assertEqual(actual.__annotations__, {"return": actual_class | None})
        self.assertEqual(expected.__annotations__, {"return": expected_class | None})
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
            for name in ("LoadEndianness", "get_default_load_endianness"):
                self.assertIs(namespace[name], getattr(module, name))

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("LoadEndianness", namespace)
            self.assertNotIn("get_default_load_endianness", namespace)

        pairs = [
            (actual_module.LoadEndianness, expected_module.LoadEndianness),
            (
                actual_module.get_default_load_endianness,
                expected_module.get_default_load_endianness,
            ),
            *zip(actual_module.LoadEndianness, expected_module.LoadEndianness),
        ]
        for actual, expected in pairs:
            self.assertIs(copy.copy(actual), actual)
            self.assertIs(copy.copy(expected), expected)
            self.assertIs(copy.deepcopy(actual), actual)
            self.assertIs(copy.deepcopy(expected), expected)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(value=actual, protocol=protocol):
                    self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                    self.assertIs(
                        pickle.loads(pickle.dumps(expected, protocol)),
                        expected,
                    )
                    self.assertEqual(
                        self.pickle_shape(actual, protocol),
                        self.pickle_shape(expected, protocol),
                    )

    def test_argument_errors_match_pytorch_2_13(self):
        actual_getter = torch.serialization.get_default_load_endianness
        expected_getter = reference_torch.serialization.get_default_load_endianness
        actual_class = torch.serialization.LoadEndianness
        expected_class = reference_torch.serialization.LoadEndianness
        original = expected_getter()
        reference_torch.serialization.set_default_load_endianness(None)
        try:
            cases = (
                (lambda: actual_getter(None), lambda: expected_getter(None)),
                (
                    lambda: actual_getter(None, None),
                    lambda: expected_getter(None, None),
                ),
                (
                    lambda: actual_getter(endianness=None),
                    lambda: expected_getter(endianness=None),
                ),
                (
                    lambda: actual_getter(None, endianness=None),
                    lambda: expected_getter(None, endianness=None),
                ),
                (lambda: actual_class(), lambda: expected_class()),
                (lambda: actual_class(1, 2), lambda: expected_class(1, 2)),
                (
                    lambda: actual_class(name="NATIVE"),
                    lambda: expected_class(name="NATIVE"),
                ),
                (lambda: actual_class(0), lambda: expected_class(0)),
                (
                    lambda: actual_class("NATIVE"),
                    lambda: expected_class("NATIVE"),
                ),
                (
                    lambda: actual_class["native"],
                    lambda: expected_class["native"],
                ),
            )
            for case, (actual_call, expected_call) in enumerate(cases):
                with self.subTest(case=case):
                    self.assert_error_matches(actual_call, expected_call)
                    self.assertIsNone(actual_getter())
                    self.assertIsNone(expected_getter())

            self.assertIs(actual_class(value=1), actual_class.NATIVE)
            self.assertIs(expected_class(value=1), expected_class.NATIVE)
            self.assertIsNone(actual_getter(**{}))
            self.assertIsNone(expected_getter(**{}))
        finally:
            reference_torch.serialization.set_default_load_endianness(original)

    def test_reload_and_reimport_match_pytorch_2_13(self):
        expected_getter = reference_torch.serialization.get_default_load_endianness
        original = expected_getter()
        reference_torch.serialization.set_default_load_endianness(None)
        try:
            self.assertEqual(
                self.reload_outcome(torch),
                self.reload_outcome(reference_torch),
            )
        finally:
            restored = (
                None
                if original is None
                else reference_torch.serialization.LoadEndianness[original.name]
            )
            reference_torch.serialization.set_default_load_endianness(restored)

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
        for name in ("set_default_load_endianness", "save", "load"):
            with self.subTest(name=name):
                self.assertTrue(hasattr(expected_module, name))
                self.assertFalse(hasattr(actual_module, name))
                self.assertNotIn(name, actual_module.__all__)

        for name in ("LoadEndianness", "get_default_load_endianness"):
            with self.subTest(top_level_name=name):
                self.assertFalse(hasattr(reference_torch, name))
                self.assertFalse(hasattr(torch, name))
                self.assertNotIn(name, torch.__all__)
        for name in ("save", "load"):
            with self.subTest(top_level_name=name):
                self.assertTrue(hasattr(reference_torch, name))
                self.assertFalse(hasattr(torch, name))
                self.assertNotIn(name, torch.__all__)


if __name__ == "__main__":
    unittest.main()
