import copy
import importlib
import inspect
import mmap
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


HAS_MMAP_FLAGS = (
    sys.platform != "win32"
    and hasattr(mmap, "MAP_PRIVATE")
    and hasattr(mmap, "MAP_SHARED")
)


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class SerializationSetDefaultMmapOptionsReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "serialization mmap option differentials require pinned "
                "PyTorch 2.13.0"
            )

    def setUp(self):
        self.original = {}
        if HAS_MMAP_FLAGS:
            for module in (torch, reference_torch):
                self.original[module] = (
                    module.serialization.get_default_mmap_options()
                )
                module.serialization.set_default_mmap_options(mmap.MAP_PRIVATE)

    def tearDown(self):
        if HAS_MMAP_FLAGS:
            for module, value in self.original.items():
                module.serialization.set_default_mmap_options(value)

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

    def mutation_outcome(self, module):
        serialization = module.serialization
        setter = serialization.set_default_mmap_options
        getter = serialization.get_default_mmap_options
        outcomes = []

        for value in (
            mmap.MAP_PRIVATE,
            mmap.MAP_SHARED,
            float(mmap.MAP_PRIVATE),
            float(mmap.MAP_SHARED),
        ):
            setter(mmap.MAP_PRIVATE)
            before = getter()
            context = setter(flags=value)
            immediate = getter()
            entered = context.__enter__()
            during = getter()
            exited = context.__exit__(None, None, None)
            after = getter()
            outcomes.append(
                (
                    type(context) is setter,
                    context.prev is before,
                    immediate is value,
                    entered is None,
                    during is value,
                    exited is None,
                    after is before,
                )
            )
        return outcomes

    def context_outcome(self, module):
        serialization = module.serialization
        setter = serialization.set_default_mmap_options
        getter = serialization.get_default_mmap_options
        setter(mmap.MAP_PRIVATE)
        states = [getter()]

        outer = setter(mmap.MAP_SHARED)
        states.append(getter())
        with outer as outer_value:
            states.extend((outer_value, getter()))
            with setter(mmap.MAP_PRIVATE) as inner_value:
                states.extend((inner_value, getter()))
            states.append(getter())
        states.append(getter())

        marker = RuntimeError("context body failed")
        try:
            with setter(mmap.MAP_SHARED):
                states.append(getter())
                raise marker
        except RuntimeError as error:
            propagated = error is marker
        else:
            propagated = False
        states.append(getter())
        return states, propagated

    def reload_outcome(self, module):
        serialization = module.serialization
        old_setter = serialization.set_default_mmap_options
        old_getter = serialization.get_default_mmap_options
        context = old_setter(mmap.MAP_SHARED)

        reloaded = importlib.reload(serialization)
        new_setter = serialization.set_default_mmap_options
        state_after_reload = (
            old_getter() == mmap.MAP_SHARED,
            serialization.get_default_mmap_options() == mmap.MAP_SHARED,
        )
        exit_result = context.__exit__(None, None, None)
        state_after_exit = serialization.get_default_mmap_options()
        new_setter(mmap.MAP_SHARED)
        old_getter_state = old_getter()
        new_setter(mmap.MAP_PRIVATE)
        return (
            reloaded is serialization,
            module.serialization is serialization,
            new_setter is not old_setter,
            state_after_reload,
            exit_result is None,
            state_after_exit == mmap.MAP_PRIVATE,
            old_getter_state == mmap.MAP_SHARED,
        )

    def reimport_outcome(self, module):
        original_module = module.serialization
        old_setter = original_module.set_default_mmap_options
        old_getter = original_module.get_default_mmap_options
        module_name = original_module.__name__
        old_setter(mmap.MAP_SHARED)

        try:
            removed = sys.modules.pop(module_name)
            replacement = importlib.import_module(module_name)
            initial = (
                old_getter() == mmap.MAP_SHARED,
                replacement.get_default_mmap_options() == mmap.MAP_SHARED,
            )
            replacement.set_default_mmap_options(mmap.MAP_PRIVATE)
            old_observation = old_getter()
            old_setter(mmap.MAP_SHARED)
            replacement_observation = replacement.get_default_mmap_options()
            replacement.set_default_mmap_options(mmap.MAP_PRIVATE)
            return (
                removed is original_module,
                replacement is not original_module,
                sys.modules[module_name] is replacement,
                module.serialization is replacement,
                initial,
                old_observation == mmap.MAP_PRIVATE,
                replacement_observation == mmap.MAP_SHARED,
            )
        finally:
            sys.modules[module_name] = original_module
            module.serialization = original_module

    @unittest.skipUnless(HAS_MMAP_FLAGS, "requires POSIX mmap flags")
    def test_immediate_mutation_and_context_behavior_match_pytorch_2_13(self):
        self.assertEqual(
            self.mutation_outcome(torch),
            self.mutation_outcome(reference_torch),
        )
        self.assertEqual(
            self.context_outcome(torch),
            self.context_outcome(reference_torch),
        )

    @unittest.skipUnless(HAS_MMAP_FLAGS, "requires POSIX mmap flags")
    def test_reload_and_reimport_behavior_match_pytorch_2_13(self):
        self.assertEqual(
            self.reload_outcome(torch),
            self.reload_outcome(reference_torch),
        )
        self.assertEqual(
            self.reimport_outcome(torch),
            self.reimport_outcome(reference_torch),
        )

    def test_signature_annotations_documentation_and_identity_match(self):
        actual_module = importlib.import_module("torch_rs.serialization")
        expected_module = importlib.import_module("torch.serialization")
        actual = actual_module.set_default_mmap_options
        expected = expected_module.set_default_mmap_options

        self.assertIs(type(actual), type(expected))
        self.assertEqual(
            str(inspect.signature(actual)), str(inspect.signature(expected))
        )
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(typing.get_type_hints(actual), typing.get_type_hints(expected))
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"),
            expected.__module__,
        )
        self.assertIs(inspect.getmodule(actual), actual_module)
        self.assertIs(inspect.getmodule(expected), expected_module)
        self.assertEqual(actual.__doc__, expected.__doc__)

        for name in ("__init__", "__enter__", "__exit__"):
            with self.subTest(name=name):
                actual_method = getattr(actual, name)
                expected_method = getattr(expected, name)
                self.assertEqual(
                    str(inspect.signature(actual_method)),
                    str(inspect.signature(expected_method)),
                )
                self.assertEqual(
                    actual_method.__annotations__,
                    expected_method.__annotations__,
                )
                self.assertEqual(
                    typing.get_type_hints(actual_method),
                    typing.get_type_hints(expected_method),
                )
                self.assertEqual(actual_method.__name__, expected_method.__name__)
                self.assertEqual(
                    actual_method.__qualname__, expected_method.__qualname__
                )
                self.assertIsNone(actual_method.__defaults__)
                self.assertEqual(
                    actual_method.__defaults__, expected_method.__defaults__
                )
                self.assertEqual(
                    actual_method.__kwdefaults__, expected_method.__kwdefaults__
                )

    def test_imports_exports_copy_and_pickle_match_pytorch_2_13(self):
        actual_module = torch.serialization
        expected_module = reference_torch.serialization
        actual = actual_module.set_default_mmap_options
        expected = expected_module.set_default_mmap_options
        supported_names = (
            "get_crc32_options",
            "set_crc32_options",
            "get_default_mmap_options",
            "set_default_mmap_options",
        )

        self.assertEqual(
            actual_module.__all__,
            [name for name in expected_module.__all__ if name in supported_names],
        )
        for module, setter in (
            (actual_module, actual),
            (expected_module, expected),
        ):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertIs(namespace["set_default_mmap_options"], setter)
            self.assertIs(copy.copy(setter), setter)
            self.assertIs(copy.deepcopy(setter), setter)

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("set_default_mmap_options", namespace)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(pickle.loads(pickle.dumps(expected, protocol)), expected)
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

    @unittest.skipUnless(HAS_MMAP_FLAGS, "requires POSIX mmap flags")
    def test_argument_and_invalid_flag_errors_match_pytorch_2_13(self):
        actual = torch.serialization.set_default_mmap_options
        expected = reference_torch.serialization.set_default_mmap_options
        actual_getter = torch.serialization.get_default_mmap_options
        expected_getter = reference_torch.serialization.get_default_mmap_options
        cases = (
            (lambda: actual(), lambda: expected()),
            (
                lambda: actual(mmap.MAP_PRIVATE, mmap.MAP_SHARED),
                lambda: expected(mmap.MAP_PRIVATE, mmap.MAP_SHARED),
            ),
            (lambda: actual(enabled=True), lambda: expected(enabled=True)),
            (
                lambda: actual(mmap.MAP_PRIVATE, flags=mmap.MAP_SHARED),
                lambda: expected(mmap.MAP_PRIVATE, flags=mmap.MAP_SHARED),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(kind="arguments", case=case):
                actual_before = actual_getter()
                expected_before = expected_getter()
                self.assert_error_matches(actual_call, expected_call)
                self.assertIs(actual_getter(), actual_before)
                self.assertIs(expected_getter(), expected_before)

        for value in (None, False, 0, "shared"):
            with self.subTest(kind="flag", value=value):
                actual_before = actual_getter()
                expected_before = expected_getter()
                self.assert_error_matches(
                    lambda value=value: actual(value),
                    lambda value=value: expected(value),
                )
                self.assertIs(actual_getter(), actual_before)
                self.assertIs(expected_getter(), expected_before)

    def test_windows_errors_and_validation_order_match_pytorch_2_13(self):
        actual_module = torch.serialization
        expected_module = reference_torch.serialization
        actual_getter = actual_module.get_default_mmap_options
        expected_getter = expected_module.get_default_mmap_options

        with mock.patch.object(actual_module, "_IS_WINDOWS", True), mock.patch.object(
            expected_module, "IS_WINDOWS", True
        ):
            for value in (getattr(mmap, "MAP_PRIVATE", None), None, "invalid"):
                with self.subTest(value=value):
                    actual_before = actual_getter()
                    expected_before = expected_getter()
                    self.assert_error_matches(
                        lambda value=value: actual_module.set_default_mmap_options(
                            value
                        ),
                        lambda value=value: expected_module.set_default_mmap_options(
                            value
                        ),
                    )
                    self.assertIs(actual_getter(), actual_before)
                    self.assertIs(expected_getter(), expected_before)


if __name__ == "__main__":
    unittest.main()
