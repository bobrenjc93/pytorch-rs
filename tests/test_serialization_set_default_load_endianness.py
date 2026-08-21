import enum
import importlib
import inspect
import sys
import threading
import types
import typing
import unittest

import torch_rs as torch


SETTER_DOC = """
    Set fallback byte order for loading files

    If byteorder mark is not present in saved checkpoint,
    this byte order is used as fallback.
    By default, it's "native" byte order.

    Args:
        endianness: the new fallback byte order
    """


class SerializationSetDefaultLoadEndiannessTests(unittest.TestCase):
    def setUp(self):
        torch.serialization.set_default_load_endianness(None)

    def tearDown(self):
        torch.serialization.set_default_load_endianness(None)

    def test_none_and_current_enum_members_update_the_getter_exactly(self):
        serialization = torch.serialization
        getter = serialization.get_default_load_endianness
        setter = serialization.set_default_load_endianness

        for value in (None, *tuple(serialization.LoadEndianness)):
            with self.subTest(value=repr(value)):
                self.assertIsNone(setter(value))
                self.assertIs(getter(), value)

        self.assertIsNone(
            setter(endianness=serialization.LoadEndianness.LITTLE)
        )
        self.assertIs(getter(), serialization.LoadEndianness.LITTLE)

    def test_updates_preserve_grad_mode(self):
        serialization = torch.serialization
        getter = serialization.get_default_load_endianness
        setter = serialization.set_default_load_endianness

        self.assertIs(torch.is_grad_enabled(), True)
        self.assertIsNone(setter(serialization.LoadEndianness.NATIVE))
        self.assertIs(getter(), serialization.LoadEndianness.NATIVE)
        self.assertIs(torch.is_grad_enabled(), True)

        with torch.no_grad():
            self.assertIs(torch.is_grad_enabled(), False)
            self.assertIsNone(setter(serialization.LoadEndianness.BIG))
            self.assertIs(getter(), serialization.LoadEndianness.BIG)
            self.assertIs(torch.is_grad_enabled(), False)

        self.assertIs(getter(), serialization.LoadEndianness.BIG)
        self.assertIs(torch.is_grad_enabled(), True)

    def test_state_is_isolated_between_fresh_threads(self):
        serialization = torch.serialization
        getter = serialization.get_default_load_endianness
        setter = serialization.set_default_load_endianness
        worker_ready = threading.Event()
        main_updated = threading.Event()
        observations = []
        errors = []

        setter(serialization.LoadEndianness.NATIVE)

        def worker():
            try:
                observations.append(getter() is None)
                setter(serialization.LoadEndianness.LITTLE)
                observations.append(
                    getter() is serialization.LoadEndianness.LITTLE
                )
                worker_ready.set()
                if not main_updated.wait(timeout=10):
                    raise RuntimeError("timed out waiting for the main thread")
                observations.append(
                    getter() is serialization.LoadEndianness.LITTLE
                )
            except BaseException as error:
                errors.append(error)
                worker_ready.set()

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(worker_ready.wait(timeout=10))
        self.assertEqual(errors, [])
        self.assertIs(getter(), serialization.LoadEndianness.NATIVE)
        setter(serialization.LoadEndianness.BIG)
        main_updated.set()
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(observations, [True, True, True])
        self.assertIs(getter(), serialization.LoadEndianness.BIG)

    def test_signature_annotations_documentation_and_module_identity(self):
        serialization = importlib.import_module("torch_rs.serialization")
        setter = serialization.set_default_load_endianness

        self.assertIs(torch.serialization, serialization)
        self.assertIs(sys.modules["torch_rs.serialization"], serialization)
        self.assertIsNone(serialization.__doc__)
        self.assertIs(type(setter), types.FunctionType)
        self.assertEqual(setter.__module__, "torch_rs.serialization")
        self.assertIs(inspect.getmodule(setter), serialization)
        self.assertIsNone(setter.__defaults__)
        self.assertIsNone(setter.__kwdefaults__)
        self.assertEqual(setter.__dict__, {})
        self.assertFalse(hasattr(setter, "__text_signature__"))
        self.assertEqual(str(inspect.signature(setter)), "(endianness)")
        self.assertEqual(setter.__annotations__, {})
        self.assertEqual(typing.get_type_hints(setter), {})
        self.assertEqual(setter.__name__, "set_default_load_endianness")
        self.assertEqual(setter.__qualname__, "set_default_load_endianness")
        self.assertEqual(
            inspect.cleandoc(setter.__doc__),
            inspect.cleandoc(SETTER_DOC),
        )
        self.assertEqual(setter.__code__.co_freevars, ())
        self.assertEqual(setter.__code__.co_cellvars, ())

    def test_invalid_values_and_call_errors_preserve_state(self):
        serialization = torch.serialization
        getter = serialization.get_default_load_endianness
        setter = serialization.set_default_load_endianness
        foreign_enum = enum.Enum(
            "LoadEndianness",
            {"NATIVE": 1, "LITTLE": 2, "BIG": 3},
        )
        setter(serialization.LoadEndianness.BIG)

        for value in (
            1,
            2,
            3,
            True,
            "native",
            "little",
            "big",
            foreign_enum.NATIVE,
            object(),
        ):
            with self.subTest(value=repr(value)):
                message = (
                    "Invalid argument type in function "
                    "set_default_load_endianness"
                )
                with self.assertRaises(TypeError) as raised:
                    setter(value)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertIs(getter(), serialization.LoadEndianness.BIG)

        cases = (
            (
                lambda: setter(),
                "set_default_load_endianness() missing 1 required positional "
                "argument: 'endianness'",
            ),
            (
                lambda: setter(None, None),
                "set_default_load_endianness() takes 1 positional argument but "
                "2 were given",
            ),
            (
                lambda: setter(enabled=True),
                "set_default_load_endianness() got an unexpected keyword "
                "argument 'enabled'",
            ),
            (
                lambda: setter(None, endianness=None),
                "set_default_load_endianness() got multiple values for argument "
                "'endianness'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertIs(getter(), serialization.LoadEndianness.BIG)

    def test_reload_and_reimport_preserve_state_and_rebind_validation(self):
        original_module = torch.serialization
        original_class = original_module.LoadEndianness
        original_member = original_class.BIG
        original_getter = original_module.get_default_load_endianness
        original_setter = original_module.set_default_load_endianness
        module_name = original_module.__name__

        original_setter(original_member)
        try:
            self.assertIs(importlib.reload(original_module), original_module)
            reloaded_class = original_module.LoadEndianness
            reloaded_getter = original_module.get_default_load_endianness
            reloaded_setter = original_module.set_default_load_endianness

            self.assertIsNot(reloaded_class, original_class)
            self.assertIsNot(reloaded_getter, original_getter)
            self.assertIsNot(reloaded_setter, original_setter)
            self.assertIs(original_getter(), original_member)
            self.assertIs(reloaded_getter(), original_member)

            for setter in (original_setter, reloaded_setter):
                with self.subTest(setter="after reload"):
                    with self.assertRaisesRegex(
                        TypeError,
                        "^Invalid argument type in function "
                        "set_default_load_endianness$",
                    ):
                        setter(original_member)
                    self.assertIs(reloaded_getter(), original_member)

            self.assertIsNone(original_setter(reloaded_class.LITTLE))
            self.assertIs(reloaded_getter(), reloaded_class.LITTLE)

            self.assertIs(sys.modules.pop(module_name), original_module)
            replacement_module = importlib.import_module(module_name)
            replacement_class = replacement_module.LoadEndianness

            self.assertIsNot(replacement_module, original_module)
            self.assertIs(torch.serialization, replacement_module)
            self.assertIsNot(replacement_class, reloaded_class)
            self.assertIs(
                replacement_module.get_default_load_endianness(),
                reloaded_class.LITTLE,
            )

            with self.assertRaisesRegex(
                TypeError,
                "^Invalid argument type in function "
                "set_default_load_endianness$",
            ):
                replacement_module.set_default_load_endianness(
                    reloaded_class.LITTLE
                )
            self.assertIsNone(
                replacement_module.set_default_load_endianness(
                    replacement_class.NATIVE
                )
            )
            self.assertIs(original_getter(), replacement_class.NATIVE)

            with self.assertRaisesRegex(
                TypeError,
                "^Invalid argument type in function "
                "set_default_load_endianness$",
            ):
                original_setter(replacement_class.NATIVE)
        finally:
            original_module.set_default_load_endianness(None)
            sys.modules[module_name] = original_module
            torch.serialization = original_module

    def test_save_and_load_remain_unsupported(self):
        serialization = torch.serialization

        self.assertIn("set_default_load_endianness", serialization.__all__)
        self.assertFalse(hasattr(torch, "set_default_load_endianness"))
        self.assertNotIn("set_default_load_endianness", torch.__all__)
        for name in ("save", "load"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(serialization, name))
                self.assertNotIn(name, serialization.__all__)


if __name__ == "__main__":
    unittest.main()
