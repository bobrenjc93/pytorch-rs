import enum
import importlib
import inspect
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
class SerializationSetDefaultLoadEndiannessReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "serialization load-endianness setter differentials require "
                "pinned PyTorch 2.13.0"
            )

    def setUp(self):
        for module in (torch, reference_torch):
            module.serialization.set_default_load_endianness(None)

    def tearDown(self):
        for module in (torch, reference_torch):
            module.serialization.set_default_load_endianness(None)

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def transition_outcome(self, module):
        serialization = module.serialization
        getter = serialization.get_default_load_endianness
        setter = serialization.set_default_load_endianness
        states = []

        for value in (None, *tuple(serialization.LoadEndianness)):
            result = setter(value)
            current = getter()
            states.append(
                (
                    result,
                    current is value,
                    None if current is None else current.name,
                    None if current is None else current.value,
                )
            )

        result = setter(endianness=serialization.LoadEndianness.LITTLE)
        states.append(
            (
                result,
                getter() is serialization.LoadEndianness.LITTLE,
                getter().name,
                getter().value,
            )
        )
        return states

    def threaded_outcome(self, module):
        serialization = module.serialization
        getter = serialization.get_default_load_endianness
        setter = serialization.set_default_load_endianness
        worker_ready = threading.Event()
        main_updated = threading.Event()
        observations = []
        errors = []

        setter(serialization.LoadEndianness.NATIVE)

        def name():
            value = getter()
            return None if value is None else value.name

        def worker():
            try:
                observations.append(name())
                observations.append(setter(serialization.LoadEndianness.LITTLE))
                observations.append(name())
                worker_ready.set()
                if not main_updated.wait(timeout=10):
                    raise RuntimeError("timed out waiting for the main thread")
                observations.append(name())
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))
                worker_ready.set()

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(worker_ready.wait(timeout=10))
        main_before = name()
        main_result = setter(serialization.LoadEndianness.BIG)
        main_after = name()
        main_updated.set()
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        return observations, errors, main_before, main_result, main_after

    def reload_outcome(self, module):
        original_module = module.serialization
        original_class = original_module.LoadEndianness
        original_member = original_class.BIG
        original_getter = original_module.get_default_load_endianness
        original_setter = original_module.set_default_load_endianness
        module_name = original_module.__name__

        original_setter(original_member)
        try:
            reloaded = importlib.reload(original_module)
            reloaded_class = reloaded.LoadEndianness
            reloaded_setter = reloaded.set_default_load_endianness
            after_reload = (
                reloaded is original_module,
                reloaded_class is not original_class,
                reloaded_setter is not original_setter,
                original_getter() is original_member,
                reloaded.get_default_load_endianness() is original_member,
            )

            stale_errors = []
            for setter in (original_setter, reloaded_setter):
                try:
                    setter(original_member)
                except Exception as error:
                    stale_errors.append((type(error).__name__, str(error)))
                else:
                    stale_errors.append(None)

            old_result = original_setter(reloaded_class.LITTLE)
            after_current_set = (
                old_result,
                original_getter() is reloaded_class.LITTLE,
                reloaded.get_default_load_endianness() is reloaded_class.LITTLE,
            )

            removed = sys.modules.pop(module_name)
            replacement = importlib.import_module(module_name)
            replacement_class = replacement.LoadEndianness
            after_reimport = (
                removed is original_module,
                replacement is not original_module,
                module.serialization is replacement,
                replacement_class is not reloaded_class,
                replacement.get_default_load_endianness()
                is reloaded_class.LITTLE,
            )

            try:
                replacement.set_default_load_endianness(reloaded_class.LITTLE)
            except Exception as error:
                replacement_stale_error = (type(error).__name__, str(error))
            else:
                replacement_stale_error = None

            replacement_result = replacement.set_default_load_endianness(
                replacement_class.NATIVE
            )
            after_replacement_set = (
                replacement_result,
                original_getter() is replacement_class.NATIVE,
                replacement.get_default_load_endianness()
                is replacement_class.NATIVE,
            )
            return (
                after_reload,
                stale_errors,
                after_current_set,
                after_reimport,
                replacement_stale_error,
                after_replacement_set,
            )
        finally:
            original_module.set_default_load_endianness(None)
            sys.modules[module_name] = original_module
            module.serialization = original_module

    def test_state_transitions_match_pytorch_2_13(self):
        self.assertEqual(
            self.transition_outcome(torch),
            self.transition_outcome(reference_torch),
        )

    def test_signature_annotations_documentation_and_identity_match(self):
        actual = torch.serialization.set_default_load_endianness
        expected = reference_torch.serialization.set_default_load_endianness

        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(actual), type(expected))
        self.assertEqual(
            str(inspect.signature(actual)),
            str(inspect.signature(expected)),
        )
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(typing.get_type_hints(actual), typing.get_type_hints(expected))
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"),
            expected.__module__,
        )
        self.assertIs(inspect.getmodule(actual), torch.serialization)
        self.assertIs(inspect.getmodule(expected), reference_torch.serialization)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )
        self.assertEqual(actual.__code__.co_freevars, expected.__code__.co_freevars)
        self.assertEqual(actual.__code__.co_cellvars, expected.__code__.co_cellvars)

    def test_invalid_values_and_call_errors_match_pytorch_2_13(self):
        actual = torch.serialization
        expected = reference_torch.serialization
        actual.set_default_load_endianness(actual.LoadEndianness.BIG)
        expected.set_default_load_endianness(expected.LoadEndianness.BIG)
        actual_foreign = enum.Enum("LoadEndianness", {"NATIVE": 1}).NATIVE
        expected_foreign = enum.Enum("LoadEndianness", {"NATIVE": 1}).NATIVE
        invalid_object = object()

        cases = (
            (
                lambda: actual.set_default_load_endianness(),
                lambda: expected.set_default_load_endianness(),
            ),
            (
                lambda: actual.set_default_load_endianness(None, None),
                lambda: expected.set_default_load_endianness(None, None),
            ),
            (
                lambda: actual.set_default_load_endianness(enabled=True),
                lambda: expected.set_default_load_endianness(enabled=True),
            ),
            (
                lambda: actual.set_default_load_endianness(
                    None,
                    endianness=None,
                ),
                lambda: expected.set_default_load_endianness(
                    None,
                    endianness=None,
                ),
            ),
            (
                lambda: actual.set_default_load_endianness(1),
                lambda: expected.set_default_load_endianness(1),
            ),
            (
                lambda: actual.set_default_load_endianness(2),
                lambda: expected.set_default_load_endianness(2),
            ),
            (
                lambda: actual.set_default_load_endianness(3),
                lambda: expected.set_default_load_endianness(3),
            ),
            (
                lambda: actual.set_default_load_endianness(True),
                lambda: expected.set_default_load_endianness(True),
            ),
            (
                lambda: actual.set_default_load_endianness("native"),
                lambda: expected.set_default_load_endianness("native"),
            ),
            (
                lambda: actual.set_default_load_endianness("little"),
                lambda: expected.set_default_load_endianness("little"),
            ),
            (
                lambda: actual.set_default_load_endianness("big"),
                lambda: expected.set_default_load_endianness("big"),
            ),
            (
                lambda: actual.set_default_load_endianness(actual_foreign),
                lambda: expected.set_default_load_endianness(expected_foreign),
            ),
            (
                lambda: actual.set_default_load_endianness(invalid_object),
                lambda: expected.set_default_load_endianness(invalid_object),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                actual_before = actual.get_default_load_endianness()
                expected_before = expected.get_default_load_endianness()
                self.assert_error_matches(actual_call, expected_call)
                self.assertIs(actual.get_default_load_endianness(), actual_before)
                self.assertIs(expected.get_default_load_endianness(), expected_before)

    def test_thread_context_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.threaded_outcome(torch),
            self.threaded_outcome(reference_torch),
        )

    def test_reload_and_reimport_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_outcome(torch),
            self.reload_outcome(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
