import copy
import importlib
import inspect
import pickle
import pickletools
import sys
import types
import typing
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class SerializationSafeGlobalsReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "serialization safe-globals differentials require pinned "
                "PyTorch 2.13.0"
            )

    def setUp(self):
        self.original_safe_globals = {
            module: module.serialization.get_safe_globals()
            for module in (torch, reference_torch)
        }
        for module in self.original_safe_globals:
            module.serialization.clear_safe_globals()

    def tearDown(self):
        for module, safe_globals in self.original_safe_globals.items():
            module.serialization.clear_safe_globals()
            module.serialization.add_safe_globals(safe_globals)

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

    def assert_global_set(self, module, *expected):
        observed = module.serialization.get_safe_globals()
        self.assertEqual(set(observed), set(expected))
        for value in expected:
            self.assertTrue(any(item is value for item in observed))

    def state_outcome(self, module, shared_objects):
        A, B, C, first, second = shared_objects
        serialization = module.serialization
        outcomes = []

        serialization.clear_safe_globals()
        outcomes.append(serialization.get_safe_globals() == [])
        outcomes.append(
            serialization.get_safe_globals()
            is not serialization.get_safe_globals()
        )
        copy_of_empty = serialization.get_safe_globals()
        copy_of_empty.append(A)
        outcomes.append(serialization.get_safe_globals() == [])

        outcomes.append(serialization.add_safe_globals([A, B, A]) is None)
        outcomes.append(set(serialization.get_safe_globals()) == {A, B})
        observed = serialization.get_safe_globals()
        observed.append(C)
        outcomes.append(set(serialization.get_safe_globals()) == {A, B})

        serialization.clear_safe_globals()
        outcomes.append(serialization.add_safe_globals("abc") is None)
        outcomes.append(set(serialization.get_safe_globals()) == {"a", "b", "c"})

        serialization.clear_safe_globals()
        outcomes.append(serialization.add_safe_globals({"x": A}) is None)
        outcomes.append(serialization.get_safe_globals() == ["x"])

        serialization.clear_safe_globals()
        outcomes.append(
            serialization.add_safe_globals(value for value in (A, B)) is None
        )
        outcomes.append(set(serialization.get_safe_globals()) == {A, B})

        serialization.clear_safe_globals()
        outcomes.append(serialization.add_safe_globals([first, second]) is None)
        observed_equal = serialization.get_safe_globals()
        outcomes.append(len(observed_equal) == 1)
        outcomes.append(observed_equal[0] is first)
        outcomes.append(serialization.add_safe_globals([second]) is None)
        observed_equal = serialization.get_safe_globals()
        outcomes.append(len(observed_equal) == 1)
        outcomes.append(observed_equal[0] is first)

        tuple_entries = (
            (A, "custom.A"),
            (A,),
            (A, "custom.A", "extra"),
            ("not callable", "custom.name"),
            (A, 123),
        )
        for entry in tuple_entries:
            serialization.clear_safe_globals()
            outcomes.append(serialization.add_safe_globals([entry]) is None)
            outcomes.append(serialization.get_safe_globals() == [entry])

        serialization.clear_safe_globals()
        return outcomes

    def context_outcome(self, module, shared_objects):
        A, B, C, _, _ = shared_objects
        serialization = module.serialization
        outcomes = []

        serialization.clear_safe_globals()
        context = serialization.safe_globals([A, B, A])
        outcomes.append(context.__dict__ == {"safe_globals": [A, B, A]})
        outcomes.append(context.__enter__() is None)
        outcomes.append(set(serialization.get_safe_globals()) == {A, B})
        outcomes.append(context.__exit__(None, None, None) is None)
        outcomes.append(serialization.get_safe_globals() == [])

        with serialization.safe_globals([A]):
            outcomes.append(set(serialization.get_safe_globals()) == {A})
            with serialization.safe_globals([A, B]):
                outcomes.append(set(serialization.get_safe_globals()) == {A, B})
            outcomes.append(serialization.get_safe_globals() == [])
        outcomes.append(serialization.get_safe_globals() == [])

        serialization.add_safe_globals([A])
        with serialization.safe_globals([B]):
            outcomes.append(set(serialization.get_safe_globals()) == {A, B})
            serialization.add_safe_globals([C])
            outcomes.append(set(serialization.get_safe_globals()) == {A, B, C})
        outcomes.append(set(serialization.get_safe_globals()) == {A, C})

        serialization.clear_safe_globals()
        try:
            with serialization.safe_globals([A]):
                outcomes.append(set(serialization.get_safe_globals()) == {A})
                raise RuntimeError("boom")
        except RuntimeError:
            outcomes.append(serialization.get_safe_globals() == [])
        else:
            outcomes.append(False)

        return outcomes

    def reload_outcome(self, module, shared_objects):
        A, B, C, _, _ = shared_objects
        original_module = module.serialization
        original_getter = original_module.get_safe_globals
        original_adder = original_module.add_safe_globals
        original_clear = original_module.clear_safe_globals
        original_context_class = original_module.safe_globals
        module_name = original_module.__name__
        outcomes = []

        original_adder([A])
        try:
            reloaded = importlib.reload(original_module)
            outcomes.extend(
                (
                    reloaded is original_module,
                    module.serialization is original_module,
                    original_getter is not original_module.get_safe_globals,
                    original_context_class is not original_module.safe_globals,
                    set(original_getter()) == {A},
                    set(original_module.get_safe_globals()) == {A},
                )
            )

            original_module.add_safe_globals([B])
            outcomes.append(set(original_getter()) == {A, B})

            removed_module = sys.modules.pop(module_name)
            replacement_module = importlib.import_module(module_name)
            outcomes.extend(
                (
                    removed_module is original_module,
                    replacement_module is not original_module,
                    sys.modules[module_name] is replacement_module,
                    module.serialization is replacement_module,
                    set(original_getter()) == {A, B},
                    set(replacement_module.get_safe_globals()) == {A, B},
                )
            )

            replacement_module.add_safe_globals([C])
            outcomes.extend(
                (
                    set(original_getter()) == {A, B, C},
                    set(replacement_module.get_safe_globals()) == {A, B, C},
                    original_clear() is None,
                    original_getter() == [],
                    replacement_module.get_safe_globals() == [],
                )
            )
        finally:
            sys.modules[module_name] = original_module
            module.serialization = original_module

        return outcomes

    def test_state_and_context_behavior_match_pytorch_2_13(self):
        class A:
            pass

        class B:
            pass

        class C:
            pass

        class Eq:
            def __init__(self, label):
                self.label = label

            def __hash__(self):
                return 1

            def __eq__(self, other):
                return isinstance(other, Eq)

        shared_objects = (A, B, C, Eq("first"), Eq("second"))

        self.assertEqual(
            self.state_outcome(torch, shared_objects),
            self.state_outcome(reference_torch, shared_objects),
        )
        self.assertTrue(all(self.state_outcome(torch, shared_objects)))

        for module in (torch, reference_torch):
            module.serialization.clear_safe_globals()
        self.assertEqual(
            self.context_outcome(torch, shared_objects),
            self.context_outcome(reference_torch, shared_objects),
        )
        self.assertTrue(all(self.context_outcome(torch, shared_objects)))

    def test_signature_annotations_documentation_and_identity_match(self):
        actual_module = importlib.import_module("torch_rs.serialization")
        expected_module = importlib.import_module("torch.serialization")
        function_names = (
            "clear_safe_globals",
            "get_safe_globals",
            "add_safe_globals",
        )

        self.assertIs(torch.serialization, actual_module)
        self.assertIs(reference_torch.serialization, expected_module)
        for name in function_names:
            with self.subTest(name=name):
                actual = getattr(actual_module, name)
                expected = getattr(expected_module, name)
                self.assertIs(type(actual), types.FunctionType)
                self.assertIs(type(expected), types.FunctionType)
                self.assertEqual(
                    str(inspect.signature(actual)),
                    str(inspect.signature(expected)),
                )
                self.assertEqual(actual.__annotations__, expected.__annotations__)
                self.assertEqual(
                    typing.get_type_hints(actual),
                    typing.get_type_hints(expected),
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

        actual_class = actual_module.safe_globals
        expected_class = expected_module.safe_globals
        self.assertIs(type(actual_class), type(expected_class))
        self.assertEqual(
            str(inspect.signature(actual_class)),
            str(inspect.signature(expected_class)),
        )
        self.assertEqual(actual_class.__annotations__, expected_class.__annotations__)
        self.assertEqual(actual_class.__name__, expected_class.__name__)
        self.assertEqual(actual_class.__qualname__, expected_class.__qualname__)
        self.assertEqual(
            actual_class.__module__.replace("torch_rs", "torch"),
            expected_class.__module__,
        )
        self.assertIs(inspect.getmodule(actual_class), actual_module)
        self.assertIs(inspect.getmodule(expected_class), expected_module)
        self.assertEqual(actual_class.__doc__, expected_class.__doc__)
        self.assertEqual(
            hasattr(actual_class, "__text_signature__"),
            hasattr(expected_class, "__text_signature__"),
        )

    def test_imports_exports_copy_and_pickle_match_the_supported_scope(self):
        actual_module = torch.serialization
        expected_module = reference_torch.serialization
        supported_names = (
            "LoadEndianness",
            "get_crc32_options",
            "set_crc32_options",
            "get_default_load_endianness",
            "set_default_load_endianness",
            "get_default_mmap_options",
            "set_default_mmap_options",
            "clear_safe_globals",
            "get_safe_globals",
            "add_safe_globals",
            "safe_globals",
        )

        self.assertEqual(
            actual_module.__all__,
            [name for name in expected_module.__all__ if name in supported_names],
        )

        actual_direct_import = {}
        expected_direct_import = {}
        exec(
            "from torch_rs.serialization import "
            "clear_safe_globals, get_safe_globals, add_safe_globals, safe_globals",
            actual_direct_import,
        )
        exec(
            "from torch.serialization import "
            "clear_safe_globals, get_safe_globals, add_safe_globals, safe_globals",
            expected_direct_import,
        )
        for name in (
            "clear_safe_globals",
            "get_safe_globals",
            "add_safe_globals",
            "safe_globals",
        ):
            self.assertIs(actual_direct_import[name], getattr(actual_module, name))
            self.assertIs(expected_direct_import[name], getattr(expected_module, name))

        for module in (actual_module, expected_module):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            for name in supported_names:
                self.assertIs(namespace[name], getattr(module, name))

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            for name in supported_names:
                self.assertNotIn(name, namespace)

        for name in supported_names:
            actual = getattr(actual_module, name)
            expected = getattr(expected_module, name)
            self.assertIs(copy.copy(actual), actual)
            self.assertIs(copy.copy(expected), expected)
            self.assertIs(copy.deepcopy(actual), actual)
            self.assertIs(copy.deepcopy(expected), expected)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(name=name, protocol=protocol):
                    self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                    self.assertIs(
                        pickle.loads(pickle.dumps(expected, protocol)),
                        expected,
                    )
                    self.assertEqual(
                        self.pickle_shape(actual, protocol),
                        self.pickle_shape(expected, protocol),
                    )

    def test_argument_errors_invalid_iterables_and_state_preservation_match(self):
        actual = torch.serialization
        expected = reference_torch.serialization

        class A:
            pass

        class B:
            pass

        actual.add_safe_globals([A])
        expected.add_safe_globals([A])
        cases = (
            (lambda: actual.get_safe_globals(None), lambda: expected.get_safe_globals(None)),
            (
                lambda: actual.get_safe_globals(enabled=True),
                lambda: expected.get_safe_globals(enabled=True),
            ),
            (
                lambda: actual.clear_safe_globals(None),
                lambda: expected.clear_safe_globals(None),
            ),
            (
                lambda: actual.clear_safe_globals(enabled=True),
                lambda: expected.clear_safe_globals(enabled=True),
            ),
            (lambda: actual.add_safe_globals(), lambda: expected.add_safe_globals()),
            (
                lambda: actual.add_safe_globals([], []),
                lambda: expected.add_safe_globals([], []),
            ),
            (
                lambda: actual.add_safe_globals(enabled=True),
                lambda: expected.add_safe_globals(enabled=True),
            ),
            (
                lambda: actual.add_safe_globals([], safe_globals=[B]),
                lambda: expected.add_safe_globals([], safe_globals=[B]),
            ),
            (lambda: actual.add_safe_globals(None), lambda: expected.add_safe_globals(None)),
            (lambda: actual.add_safe_globals(1), lambda: expected.add_safe_globals(1)),
            (
                lambda: actual.add_safe_globals(object()),
                lambda: expected.add_safe_globals(object()),
            ),
            (
                lambda: actual.add_safe_globals([[B]]),
                lambda: expected.add_safe_globals([[B]]),
            ),
            (
                lambda: actual.add_safe_globals([{B}]),
                lambda: expected.add_safe_globals([{B}]),
            ),
            (
                lambda: actual.add_safe_globals([B, {"k": B}]),
                lambda: expected.add_safe_globals([B, {"k": B}]),
            ),
            (lambda: actual.safe_globals(), lambda: expected.safe_globals()),
            (
                lambda: actual.safe_globals([], []),
                lambda: expected.safe_globals([], []),
            ),
            (
                lambda: actual.safe_globals(enabled=True),
                lambda: expected.safe_globals(enabled=True),
            ),
            (
                lambda: actual.safe_globals(None).__enter__(),
                lambda: expected.safe_globals(None).__enter__(),
            ),
            (
                lambda: actual.safe_globals([[B]]).__enter__(),
                lambda: expected.safe_globals([[B]]).__enter__(),
            ),
        )

        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                actual_before = actual.get_safe_globals()
                expected_before = expected.get_safe_globals()
                self.assert_error_matches(actual_call, expected_call)
                self.assertEqual(actual.get_safe_globals(), actual_before)
                self.assertEqual(expected.get_safe_globals(), expected_before)

        self.assertIsNone(actual.add_safe_globals(safe_globals=[B]))
        self.assertIsNone(expected.add_safe_globals(safe_globals=[B]))
        self.assert_global_set(torch, A, B)
        self.assert_global_set(reference_torch, A, B)

    def test_reload_and_reimport_behavior_matches_pytorch_2_13(self):
        class A:
            pass

        class B:
            pass

        class C:
            pass

        shared_objects = (A, B, C, object(), object())
        self.assertEqual(
            self.reload_outcome(torch, shared_objects),
            self.reload_outcome(reference_torch, shared_objects),
        )
        self.assertTrue(all(self.reload_outcome(torch, shared_objects)))

    def test_save_load_and_checkpoint_scanning_remain_unsupported(self):
        actual_module = torch.serialization
        expected_module = reference_torch.serialization
        actual_public = {
            name for name in vars(actual_module) if not name.startswith("_")
        }
        supported_names = {
            "LoadEndianness",
            "get_crc32_options",
            "set_crc32_options",
            "get_default_load_endianness",
            "set_default_load_endianness",
            "get_default_mmap_options",
            "set_default_mmap_options",
            "clear_safe_globals",
            "get_safe_globals",
            "add_safe_globals",
            "safe_globals",
        }

        self.assertEqual(actual_public, supported_names)
        unsupported = set(expected_module.__all__) - actual_public
        self.assertTrue(
            {"save", "load", "get_unsafe_globals_in_checkpoint"}.issubset(
                unsupported
            )
        )
        for name in unsupported:
            with self.subTest(name=name):
                self.assertFalse(hasattr(actual_module, name))

        for name in ("save", "load"):
            with self.subTest(top_level_name=name):
                self.assertTrue(hasattr(reference_torch, name))
                self.assertFalse(hasattr(torch, name))
                self.assertNotIn(name, torch.__all__)
        for name in supported_names:
            with self.subTest(top_level_name=name):
                self.assertFalse(hasattr(reference_torch, name))
                self.assertFalse(hasattr(torch, name))
                self.assertNotIn(name, torch.__all__)


if __name__ == "__main__":
    unittest.main()
