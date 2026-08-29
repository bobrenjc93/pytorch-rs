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


class SafeClass:
    pass


class OtherSafeClass:
    pass


def safe_function():
    pass


class EquivalentGlobal:
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, EquivalentGlobal)

    def __hash__(self):
        return 1


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
        self.original = {
            module: module.serialization.get_safe_globals()
            for module in (torch, reference_torch)
        }
        for module in self.original:
            module.serialization.clear_safe_globals()

    def tearDown(self):
        for module, safe_globals in self.original.items():
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

    def assert_same_safe_state(self, actual_module, expected_module, expected_items):
        actual = actual_module.serialization.get_safe_globals()
        expected = expected_module.serialization.get_safe_globals()
        self.assertEqual(set(actual), set(expected_items))
        self.assertEqual(set(expected), set(expected_items))
        self.assertEqual(len(actual), len(set(expected_items)))
        self.assertEqual(len(expected), len(set(expected_items)))
        self.assertEqual(set(actual), set(expected))

    def test_metadata_imports_copy_and_pickle_match_pytorch_2_13(self):
        actual_module = importlib.import_module("torch_rs.serialization")
        expected_module = importlib.import_module("torch.serialization")
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

        self.assertIs(torch.serialization, actual_module)
        self.assertIs(reference_torch.serialization, expected_module)
        self.assertIsNone(actual_module.__doc__)
        self.assertEqual(actual_module.__doc__, expected_module.__doc__)
        self.assertEqual(
            actual_module.__all__,
            [name for name in expected_module.__all__ if name in supported_names],
        )

        for name in ("clear_safe_globals", "get_safe_globals", "add_safe_globals"):
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

        actual_context = actual_module.safe_globals
        expected_context = expected_module.safe_globals
        self.assertIs(type(actual_context), type(expected_context))
        self.assertEqual(
            str(inspect.signature(actual_context)),
            str(inspect.signature(expected_context)),
        )
        self.assertEqual(
            actual_context.__annotations__,
            expected_context.__annotations__,
        )
        self.assertEqual(actual_context.__name__, expected_context.__name__)
        self.assertEqual(actual_context.__qualname__, expected_context.__qualname__)
        self.assertEqual(
            actual_context.__module__.replace("torch_rs", "torch"),
            expected_context.__module__,
        )
        self.assertIs(inspect.getmodule(actual_context), actual_module)
        self.assertIs(inspect.getmodule(expected_context), expected_module)
        self.assertEqual(actual_context.__doc__, expected_context.__doc__)
        self.assertEqual(
            actual_context.__bases__[0].__name__,
            expected_context.__bases__[0].__name__,
        )
        self.assertEqual(
            actual_context.__bases__[0].__module__.replace("torch_rs", "torch"),
            expected_context.__bases__[0].__module__,
        )

        for method_name in ("__init__", "__enter__", "__exit__"):
            with self.subTest(method=method_name):
                actual_method = getattr(actual_context, method_name)
                expected_method = getattr(expected_context, method_name)
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
                    actual_method.__kwdefaults__,
                    expected_method.__kwdefaults__,
                )
                self.assertEqual(actual_method.__dict__, expected_method.__dict__)
                self.assertEqual(
                    actual_method.__module__.replace("torch_rs", "torch"),
                    expected_method.__module__,
                )
                self.assertEqual(
                    actual_method.__qualname__,
                    expected_method.__qualname__,
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
            for name in (
                "clear_safe_globals",
                "get_safe_globals",
                "add_safe_globals",
                "safe_globals",
            ):
                self.assertFalse(hasattr(module, name))
                self.assertNotIn(name, module.__all__)
                self.assertNotIn(name, namespace)

        for name in (
            "clear_safe_globals",
            "get_safe_globals",
            "add_safe_globals",
            "safe_globals",
        ):
            actual = getattr(actual_module, name)
            expected = getattr(expected_module, name)
            self.assertIs(copy.copy(actual), actual)
            self.assertIs(copy.copy(expected), expected)
            self.assertIs(copy.deepcopy(actual), actual)
            self.assertIs(copy.deepcopy(expected), expected)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(name=name, protocol=protocol):
                    self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                    self.assertIs(pickle.loads(pickle.dumps(expected, protocol)), expected)
                    self.assertEqual(
                        self.pickle_shape(actual, protocol),
                        self.pickle_shape(expected, protocol),
                    )

    def test_global_state_snapshots_and_addition_match_pytorch_2_13(self):
        actual = torch.serialization
        expected = reference_torch.serialization

        for module in (torch, reference_torch):
            first = module.serialization.get_safe_globals()
            second = module.serialization.get_safe_globals()
            self.assertEqual(first, [])
            self.assertEqual(second, [])
            self.assertIsNot(first, second)
            first.append(SafeClass)
            self.assertEqual(module.serialization.get_safe_globals(), [])

        for make_iterable in (
            lambda: [SafeClass, OtherSafeClass],
            lambda: (SafeClass, OtherSafeClass),
            lambda: iter([SafeClass, OtherSafeClass]),
            lambda: (item for item in [SafeClass, OtherSafeClass]),
        ):
            with self.subTest(iterable=make_iterable):
                for module in (torch, reference_torch):
                    module.serialization.clear_safe_globals()
                    self.assertIsNone(
                        module.serialization.add_safe_globals(make_iterable())
                    )
                self.assert_same_safe_state(
                    torch,
                    reference_torch,
                    {SafeClass, OtherSafeClass},
                )

        for module in (torch, reference_torch):
            module.serialization.clear_safe_globals()
            module.serialization.add_safe_globals(
                [
                    (SafeClass, "custom.safe"),
                    (SafeClass, "custom.safe"),
                    (SafeClass, "custom.other"),
                    SafeClass,
                ]
            )
        self.assert_same_safe_state(
            torch,
            reference_torch,
            {
                (SafeClass, "custom.safe"),
                (SafeClass, "custom.other"),
                SafeClass,
            },
        )

        twin_a = type("Twin", (), {})
        twin_b = type("Twin", (), {})
        twin_b.__module__ = twin_a.__module__
        for module in (torch, reference_torch):
            module.serialization.clear_safe_globals()
            module.serialization.add_safe_globals([twin_a, twin_b, twin_a])
        self.assertEqual(
            {id(item) for item in actual.get_safe_globals()},
            {id(twin_a), id(twin_b)},
        )
        self.assertEqual(
            {id(item) for item in expected.get_safe_globals()},
            {id(twin_a), id(twin_b)},
        )

        first = EquivalentGlobal("first")
        second = EquivalentGlobal("second")
        for module in (torch, reference_torch):
            module.serialization.clear_safe_globals()
            module.serialization.add_safe_globals([first, second])
            observed = module.serialization.get_safe_globals()
            self.assertEqual(len(observed), 1)
            self.assertIs(observed[0], first)

        for module in (torch, reference_torch):
            module.serialization.clear_safe_globals()
            module.serialization.add_safe_globals("xyx")
        self.assert_same_safe_state(torch, reference_torch, {"x", "y"})

    def test_context_copy_pickle_nesting_and_removal_match_pytorch_2_13(self):
        for module in (torch, reference_torch):
            context_class = module.serialization.safe_globals
            context = context_class([SafeClass, safe_function])
            shallow = copy.copy(context)
            deep = copy.deepcopy(context)
            self.assertIs(type(context), context_class)
            self.assertEqual(
                context.__dict__,
                {"safe_globals": [SafeClass, safe_function]},
            )
            self.assertIsNot(shallow, context)
            self.assertIs(shallow.safe_globals, context.safe_globals)
            self.assertIsNot(deep, context)
            self.assertIsNot(deep.safe_globals, context.safe_globals)
            self.assertEqual(deep.safe_globals, context.safe_globals)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                rehydrated = pickle.loads(pickle.dumps(context, protocol))
                self.assertIs(type(rehydrated), context_class)
                self.assertEqual(rehydrated.__dict__, context.__dict__)

        for module in (torch, reference_torch):
            module.serialization.clear_safe_globals()
            with module.serialization.safe_globals([SafeClass]) as outer:
                self.assertIsNone(outer)
                self.assertEqual(
                    set(module.serialization.get_safe_globals()),
                    {SafeClass},
                )
                with module.serialization.safe_globals([OtherSafeClass]) as inner:
                    self.assertIsNone(inner)
                    self.assertEqual(
                        set(module.serialization.get_safe_globals()),
                        {SafeClass, OtherSafeClass},
                    )
                self.assertEqual(
                    set(module.serialization.get_safe_globals()),
                    {SafeClass},
                )
            self.assertEqual(module.serialization.get_safe_globals(), [])

            module.serialization.add_safe_globals([SafeClass])
            with module.serialization.safe_globals([SafeClass, OtherSafeClass]):
                self.assertEqual(
                    set(module.serialization.get_safe_globals()),
                    {SafeClass, OtherSafeClass},
                )
            self.assertEqual(module.serialization.get_safe_globals(), [])

            values = [SafeClass]
            context = module.serialization.safe_globals(values)
            context.__enter__()
            values[:] = [OtherSafeClass]
            context.__exit__(None, None, None)
            self.assertEqual(set(module.serialization.get_safe_globals()), {SafeClass})

    def test_invalid_iterables_and_argument_errors_match_pytorch_2_13(self):
        actual = torch.serialization
        expected = reference_torch.serialization
        callable_pairs = (
            (lambda module: module.clear_safe_globals(None), lambda module: None),
            (
                lambda module: module.clear_safe_globals(None, None),
                lambda module: None,
            ),
            (
                lambda module: module.clear_safe_globals(enabled=True),
                lambda module: None,
            ),
            (lambda module: module.get_safe_globals(None), lambda module: None),
            (
                lambda module: module.get_safe_globals(None, None),
                lambda module: None,
            ),
            (
                lambda module: module.get_safe_globals(enabled=True),
                lambda module: None,
            ),
            (lambda module: module.add_safe_globals(), lambda module: None),
            (
                lambda module: module.add_safe_globals([], []),
                lambda module: None,
            ),
            (
                lambda module: module.add_safe_globals(enabled=True),
                lambda module: None,
            ),
            (
                lambda module: module.add_safe_globals([], safe_globals=[]),
                lambda module: None,
            ),
            (lambda module: module.safe_globals(), lambda module: None),
            (lambda module: module.safe_globals([], []), lambda module: None),
            (lambda module: module.safe_globals(enabled=True), lambda module: None),
            (
                lambda module: module.safe_globals([], safe_globals=[]),
                lambda module: None,
            ),
        )
        for case, (make_call, _) in enumerate(callable_pairs):
            with self.subTest(argument_case=case):
                for module in (actual, expected):
                    module.clear_safe_globals()
                    module.add_safe_globals([SafeClass])
                self.assert_error_matches(
                    lambda make_call=make_call: make_call(actual),
                    lambda make_call=make_call: make_call(expected),
                )
                self.assert_same_safe_state(
                    torch,
                    reference_torch,
                    {SafeClass},
                )

        for value in (None, 1, True, object()):
            with self.subTest(invalid_iterable=repr(value)):
                for module in (actual, expected):
                    module.clear_safe_globals()
                    module.add_safe_globals([SafeClass])
                self.assert_error_matches(
                    lambda value=value: actual.add_safe_globals(value),
                    lambda value=value: expected.add_safe_globals(value),
                )
                self.assert_same_safe_state(torch, reference_torch, {SafeClass})

                actual_context = actual.safe_globals(value)
                expected_context = expected.safe_globals(value)
                self.assert_error_matches(
                    actual_context.__enter__,
                    expected_context.__enter__,
                )
                self.assert_same_safe_state(torch, reference_torch, {SafeClass})
                self.assert_error_matches(
                    lambda: actual_context.__exit__(None, None, None),
                    lambda: expected_context.__exit__(None, None, None),
                )
                self.assert_same_safe_state(torch, reference_torch, {SafeClass})

        for module in (actual, expected):
            module.clear_safe_globals()
            module.add_safe_globals([SafeClass])
        self.assert_error_matches(
            lambda: actual.add_safe_globals([[]]),
            lambda: expected.add_safe_globals([[]]),
        )
        self.assert_same_safe_state(torch, reference_torch, {SafeClass})

        class MarkerError(Exception):
            pass

        def generator():
            yield OtherSafeClass
            raise MarkerError("boom")

        self.assert_error_matches(
            lambda: actual.add_safe_globals(generator()),
            lambda: expected.add_safe_globals(generator()),
        )
        self.assert_same_safe_state(torch, reference_torch, {SafeClass})

    def test_reload_reimport_and_unsupported_scope_match_pytorch_2_13(self):
        def reload_outcome(module):
            original_module = module.serialization
            original_getter = original_module.get_safe_globals
            original_adder = original_module.add_safe_globals
            original_clearer = original_module.clear_safe_globals
            original_context_class = original_module.safe_globals
            module_name = original_module.__name__

            original_clearer()
            original_adder([SafeClass])
            try:
                reloaded = importlib.reload(original_module)
                after_reload = (
                    reloaded is original_module,
                    original_getter is not reloaded.get_safe_globals,
                    original_adder is not reloaded.add_safe_globals,
                    original_clearer is not reloaded.clear_safe_globals,
                    original_context_class is not reloaded.safe_globals,
                    set(original_getter()) == {SafeClass},
                    set(reloaded.get_safe_globals()) == {SafeClass},
                )
                reloaded.add_safe_globals([OtherSafeClass])
                with original_context_class([safe_function]):
                    old_context_state = set(reloaded.get_safe_globals()) == {
                        SafeClass,
                        OtherSafeClass,
                        safe_function,
                    }
                after_old_context = set(original_getter()) == {
                    SafeClass,
                    OtherSafeClass,
                }

                removed = sys.modules.pop(module_name)
                replacement_module = importlib.import_module(module_name)
                after_reimport = (
                    removed is original_module,
                    replacement_module is not original_module,
                    sys.modules[module_name] is replacement_module,
                    module.serialization is replacement_module,
                    set(original_getter()) == {SafeClass, OtherSafeClass},
                    set(replacement_module.get_safe_globals())
                    == {SafeClass, OtherSafeClass},
                )
                replacement_module.add_safe_globals([safe_function])
                after_replacement_add = set(original_getter()) == {
                    SafeClass,
                    OtherSafeClass,
                    safe_function,
                }
                original_clearer()
                after_original_clear = (
                    reloaded.get_safe_globals(),
                    replacement_module.get_safe_globals(),
                )
                return (
                    after_reload,
                    old_context_state,
                    after_old_context,
                    after_reimport,
                    after_replacement_add,
                    after_original_clear,
                )
            finally:
                sys.modules[module_name] = original_module
                module.serialization = original_module

        self.assertEqual(
            reload_outcome(torch),
            reload_outcome(reference_torch),
        )

        actual_public = {
            name for name in vars(torch.serialization) if not name.startswith("_")
        }
        unsupported = set(reference_torch.serialization.__all__) - actual_public
        self.assertTrue(
            {
                "save",
                "load",
                "get_unsafe_globals_in_checkpoint",
            }.issubset(unsupported)
        )
        for name in unsupported:
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.serialization, name))


if __name__ == "__main__":
    unittest.main()
