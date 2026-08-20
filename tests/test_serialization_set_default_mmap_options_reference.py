import importlib
import inspect
import mmap
import sys
import typing
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class SerializationSetDefaultMmapOptionsReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "serialization mmap setter differentials require pinned "
                "PyTorch 2.13.0"
            )

    def setUp(self):
        self.private = getattr(mmap, "MAP_PRIVATE", None)
        self.shared = getattr(mmap, "MAP_SHARED", None)
        self.supported = (
            sys.platform != "win32"
            and self.private is not None
            and self.shared is not None
        )
        self.original = {
            module: module.serialization.get_default_mmap_options()
            for module in (torch, reference_torch)
        }
        if self.supported:
            for module in self.original:
                module.serialization.set_default_mmap_options(self.private)

    def tearDown(self):
        if self.supported:
            for module, flags in self.original.items():
                module.serialization.set_default_mmap_options(flags)

    def require_supported(self):
        if not self.supported:
            self.skipTest("mmap.MAP_PRIVATE and mmap.MAP_SHARED are required")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def context_outcome(self, module):
        getter = module.serialization.get_default_mmap_options
        setter = module.serialization.set_default_mmap_options
        states = [getter()]

        direct = setter(self.shared)
        states.extend(
            (
                type(direct) is setter,
                direct.prev,
                getter(),
                direct.__enter__(),
                getter(),
                direct.__exit__(None, None, None),
                getter(),
            )
        )

        class MarkerError(Exception):
            pass

        try:
            with setter(self.shared) as outer_value:
                states.extend((outer_value, getter()))
                with setter(self.private) as inner_value:
                    states.extend((inner_value, getter()))
                states.append(getter())
                with setter(self.private):
                    states.append(getter())
                    raise MarkerError("boom")
        except MarkerError as error:
            states.extend((type(error).__name__, str(error), getter()))

        keyword = setter(flags=self.shared)
        states.extend((getter(), keyword.__exit__(None, None, None), getter()))
        return states

    def reload_outcome(self, module):
        original_module = module.serialization
        original_getter = original_module.get_default_mmap_options
        original_setter = original_module.set_default_mmap_options
        module_name = original_module.__name__
        outer = original_setter(self.shared)
        try:
            reloaded = importlib.reload(original_module)
            reloaded_setter = reloaded.set_default_mmap_options
            after_reload = (
                reloaded is original_module,
                reloaded_setter is not original_setter,
                original_getter(),
                reloaded.get_default_mmap_options(),
            )
            with reloaded_setter(self.private):
                reloaded_context = original_getter()
            after_reloaded_context = original_getter()

            removed = sys.modules.pop(module_name)
            replacement_module = importlib.import_module(module_name)
            after_reimport = (
                removed is original_module,
                replacement_module is not original_module,
                sys.modules[module_name] is replacement_module,
                module.serialization is replacement_module,
                original_getter(),
                replacement_module.get_default_mmap_options(),
            )
            with replacement_module.set_default_mmap_options(self.private):
                replacement_context = original_getter()
            after_replacement_context = original_getter()

            outer.__exit__(None, None, None)
            outer = None
            after_old_exit = (
                original_getter(),
                replacement_module.get_default_mmap_options(),
            )
            return (
                after_reload,
                reloaded_context,
                after_reloaded_context,
                after_reimport,
                replacement_context,
                after_replacement_context,
                after_old_exit,
            )
        finally:
            if outer is not None:
                outer.__exit__(None, None, None)
            sys.modules[module_name] = original_module
            module.serialization = original_module

    def test_signature_annotations_documentation_and_identity_match(self):
        actual = torch.serialization.set_default_mmap_options
        expected = reference_torch.serialization.set_default_mmap_options

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
        self.assertEqual(actual.__bases__, expected.__bases__)

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
                self.assertEqual(
                    actual_method.__defaults__,
                    expected_method.__defaults__,
                )
                self.assertEqual(
                    actual_method.__kwdefaults__, expected_method.__kwdefaults__
                )
                self.assertEqual(actual_method.__dict__, expected_method.__dict__)

    def test_immediate_nested_and_exceptional_behavior_matches_pytorch_2_13(self):
        self.require_supported()
        self.assertEqual(
            self.context_outcome(torch),
            self.context_outcome(reference_torch),
        )

    def test_invalid_flags_call_errors_and_windows_guard_match(self):
        self.require_supported()
        actual = torch.serialization.set_default_mmap_options
        expected = reference_torch.serialization.set_default_mmap_options
        actual_getter = torch.serialization.get_default_mmap_options
        expected_getter = reference_torch.serialization.get_default_mmap_options

        cases = (
            (lambda: actual(0), lambda: expected(0)),
            (lambda: actual(-1), lambda: expected(-1)),
            (
                lambda: actual(self.private | self.shared),
                lambda: expected(self.private | self.shared),
            ),
            (lambda: actual(False), lambda: expected(False)),
            (lambda: actual(None), lambda: expected(None)),
            (lambda: actual("invalid"), lambda: expected("invalid")),
            (lambda: actual(), lambda: expected()),
            (
                lambda: actual(self.private, self.shared),
                lambda: expected(self.private, self.shared),
            ),
            (lambda: actual(enabled=True), lambda: expected(enabled=True)),
            (
                lambda: actual(self.private, flags=self.shared),
                lambda: expected(self.private, flags=self.shared),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                actual_before = actual_getter()
                expected_before = expected_getter()
                self.assert_error_matches(actual_call, expected_call)
                self.assertIs(actual_getter(), actual_before)
                self.assertIs(expected_getter(), expected_before)

        actual_windows = torch.serialization._IS_WINDOWS
        expected_windows = reference_torch.serialization.IS_WINDOWS
        torch.serialization._IS_WINDOWS = True
        reference_torch.serialization.IS_WINDOWS = True
        try:
            self.assert_error_matches(
                lambda: actual(self.private),
                lambda: expected(self.private),
            )
            self.assertIs(actual_getter(), self.private)
            self.assertIs(expected_getter(), self.private)
        finally:
            torch.serialization._IS_WINDOWS = actual_windows
            reference_torch.serialization.IS_WINDOWS = expected_windows

    def test_reload_and_reimport_behavior_matches_pytorch_2_13(self):
        self.require_supported()
        self.assertEqual(
            self.reload_outcome(torch),
            self.reload_outcome(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
