import collections.abc
import copy
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


SAFE_GLOBALS_ANNOTATION = list[
    collections.abc.Callable | tuple[collections.abc.Callable, str]
]

ADD_DOC = """
    Marks the given globals as safe for ``weights_only`` load. For example, functions
    added to this list can be called during unpickling, classes could be instantiated
    and have state set.

    Each item in the list can either be a function/class or a tuple of the form
    (function/class, string) where string is the full path of the function/class.

    Within the serialized format, each function is identified with its full
    path as ``{__module__}.{__qualname__}``. When calling this API, you can provide this
    full path that should match the one in the checkpoint otherwise the default
    ``{fn.__module__}.{fn.__qualname__}`` will be used.

    Args:
        safe_globals (List[Union[Callable, Tuple[Callable, str]]]): list of globals to mark as safe

    Example:
        >>> # xdoctest: +SKIP("Can't torch.save(t, ...) as doctest thinks MyTensor is defined on torch.serialization")
        >>> import tempfile
        >>> class MyTensor(torch.Tensor):
        ...     pass
        >>> t = MyTensor(torch.randn(2, 3))
        >>> with tempfile.NamedTemporaryFile() as f:
        ...     torch.save(t, f.name)
        # Running `torch.load(f.name, weights_only=True)` will fail with
        # Unsupported global: GLOBAL __main__.MyTensor was not an allowed global by default.
        # Check the code and make sure MyTensor is safe to be used when loaded from an arbitrary checkpoint.
        ...     torch.serialization.add_safe_globals([MyTensor])
        ...     torch.load(f.name, weights_only=True)
        # MyTensor([[-0.5024, -1.8152, -0.5455],
        #          [-0.8234,  2.0500, -0.3657]])
    """

GET_DOC = """
    Returns the list of user-added globals that are safe for ``weights_only`` load.
    """

CLEAR_DOC = """
    Clears the list of globals that are safe for ``weights_only`` load.
    """

CONTEXT_DOC = """Context-manager that adds certain globals as safe for ``weights_only`` load.

    Args:
        safe_globals: List of globals for weights_only load.

    Example:
        >>> # xdoctest: +SKIP("Can't torch.save(t, ...) as doctest thinks MyTensor is defined on torch.serialization")
        >>> import tempfile
        >>> class MyTensor(torch.Tensor):
        ...     pass
        >>> t = MyTensor(torch.randn(2, 3))
        >>> with tempfile.NamedTemporaryFile() as f:
        ...     torch.save(t, f.name)
        # Running `torch.load(f.name, weights_only=True)` will fail with
        # Unsupported global: GLOBAL __main__.MyTensor was not an allowed global by default.
        # Check the code and make sure MyTensor is safe to be used when loaded from an arbitrary checkpoint.
        ...     with torch.serialization.safe_globals([MyTensor]):
        ...         torch.load(f.name, weights_only=True)
        # MyTensor([[-0.5024, -1.8152, -0.5455],
        #          [-0.8234,  2.0500, -0.3657]])
        >>> assert torch.serialization.get_safe_globals() == []
    """


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


class SerializationSafeGlobalsTests(unittest.TestCase):
    def setUp(self):
        self.serialization = torch.serialization
        self.original = self.serialization.get_safe_globals()
        self.serialization.clear_safe_globals()

    def tearDown(self):
        self.serialization.clear_safe_globals()
        self.serialization.add_safe_globals(self.original)

    def assert_state_is(self, expected):
        observed = self.serialization.get_safe_globals()
        self.assertEqual(set(observed), set(expected))
        self.assertEqual(len(observed), len(set(expected)))
        return observed

    def test_default_clear_getter_and_thread_visibility(self):
        getter = self.serialization.get_safe_globals
        clearer = self.serialization.clear_safe_globals
        adder = self.serialization.add_safe_globals
        self.assertEqual(getter.__code__.co_freevars, ())
        self.assertEqual(getter.__code__.co_cellvars, ())
        self.assertEqual(clearer.__code__.co_freevars, ())
        self.assertEqual(clearer.__code__.co_cellvars, ())

        self.assertIsNone(clearer())
        first = getter()
        second = getter()
        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertIsNot(first, second)
        first.append(SafeClass)
        self.assertEqual(getter(), [])

        adder([SafeClass])
        worker_ready = threading.Event()
        read_updated = threading.Event()
        observations = []
        errors = []

        def observer():
            try:
                observations.append(set(getter()) == {SafeClass})
                worker_ready.set()
                if not read_updated.wait(timeout=10):
                    raise RuntimeError("timed out waiting for safe globals update")
                observations.append(set(getter()) == {OtherSafeClass})
            except BaseException as error:
                errors.append(error)
                worker_ready.set()

        thread = threading.Thread(target=observer)
        thread.start()
        self.assertTrue(worker_ready.wait(timeout=10))
        clearer()
        adder([OtherSafeClass])
        read_updated.set()
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(observations, [True, True])
        self.assert_state_is([OtherSafeClass])

    def test_add_accepts_iterables_and_deduplicates_by_hash_equality(self):
        add = self.serialization.add_safe_globals

        for make_iterable in (
            lambda: [SafeClass, OtherSafeClass],
            lambda: (SafeClass, OtherSafeClass),
            lambda: iter([SafeClass, OtherSafeClass]),
            lambda: (item for item in [SafeClass, OtherSafeClass]),
        ):
            with self.subTest(iterable=make_iterable):
                self.serialization.clear_safe_globals()
                self.assertIsNone(add(make_iterable()))
                self.assert_state_is([SafeClass, OtherSafeClass])

        self.serialization.clear_safe_globals()
        self.assertIsNone(
            add(
                [
                    (SafeClass, "custom.safe"),
                    (SafeClass, "custom.safe"),
                    (SafeClass, "custom.other"),
                    SafeClass,
                ]
            )
        )
        self.assert_state_is(
            [
                (SafeClass, "custom.safe"),
                (SafeClass, "custom.other"),
                SafeClass,
            ]
        )

        twin_a = type("Twin", (), {})
        twin_b = type("Twin", (), {})
        twin_b.__module__ = twin_a.__module__
        self.serialization.clear_safe_globals()
        add([twin_a, twin_b, twin_a])
        observed = self.serialization.get_safe_globals()
        self.assertEqual({id(item) for item in observed}, {id(twin_a), id(twin_b)})
        self.assertEqual(len(observed), 2)

        first = EquivalentGlobal("first")
        second = EquivalentGlobal("second")
        self.serialization.clear_safe_globals()
        add([first, second])
        observed = self.serialization.get_safe_globals()
        self.assertEqual(len(observed), 1)
        self.assertIs(observed[0], first)

        self.serialization.clear_safe_globals()
        add("xyx")
        self.assert_state_is(["x", "y"])

    def test_invalid_iterables_and_unhashable_items_preserve_state(self):
        add = self.serialization.add_safe_globals
        getter = self.serialization.get_safe_globals

        for value, message in (
            (None, "'NoneType' object is not iterable"),
            (1, "'int' object is not iterable"),
            (True, "'bool' object is not iterable"),
            (object(), "'object' object is not iterable"),
        ):
            with self.subTest(value=repr(value)):
                self.serialization.clear_safe_globals()
                add([SafeClass])
                before = getter()
                with self.assertRaises(TypeError) as raised:
                    add(value)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assert_state_is(before)

        self.serialization.clear_safe_globals()
        add([SafeClass])
        with self.assertRaises(TypeError) as raised:
            add([[]])
        self.assertEqual(str(raised.exception), "unhashable type: 'list'")
        self.assertEqual(raised.exception.args, ("unhashable type: 'list'",))
        self.assert_state_is([SafeClass])

        class MarkerError(Exception):
            pass

        def generator():
            yield OtherSafeClass
            raise MarkerError("boom")

        with self.assertRaises(MarkerError) as raised:
            add(generator())
        self.assertEqual(str(raised.exception), "boom")
        self.assert_state_is([SafeClass])

    def test_contexts_add_on_enter_and_remove_by_set_on_exit(self):
        safe_globals = self.serialization.safe_globals
        getter = self.serialization.get_safe_globals
        add = self.serialization.add_safe_globals

        context = safe_globals([SafeClass, safe_function])
        self.assertIs(type(context), safe_globals)
        self.assertEqual(context.__dict__, {"safe_globals": [SafeClass, safe_function]})
        shallow = copy.copy(context)
        deep = copy.deepcopy(context)
        self.assertIsNot(shallow, context)
        self.assertIs(shallow.safe_globals, context.safe_globals)
        self.assertIsNot(deep, context)
        self.assertIsNot(deep.safe_globals, context.safe_globals)
        self.assertEqual(deep.safe_globals, context.safe_globals)
        payload = pickle.dumps(context)
        rehydrated = pickle.loads(payload)
        self.assertIs(type(rehydrated), safe_globals)
        self.assertEqual(rehydrated.__dict__, context.__dict__)

        self.assertEqual(getter(), [])
        self.assertIsNone(context.__enter__())
        self.assert_state_is([SafeClass, safe_function])
        self.assertIsNone(context.__exit__(None, None, None))
        self.assertEqual(getter(), [])

        class MarkerError(Exception):
            pass

        with safe_globals([SafeClass]) as outer:
            self.assertIsNone(outer)
            self.assert_state_is([SafeClass])
            with safe_globals([OtherSafeClass]) as inner:
                self.assertIsNone(inner)
                self.assert_state_is([SafeClass, OtherSafeClass])
            self.assert_state_is([SafeClass])
            with self.assertRaisesRegex(MarkerError, "boom"):
                with safe_globals([OtherSafeClass]):
                    self.assert_state_is([SafeClass, OtherSafeClass])
                    raise MarkerError("boom")
            self.assert_state_is([SafeClass])
        self.assertEqual(getter(), [])

        add([SafeClass])
        with safe_globals([SafeClass, OtherSafeClass]):
            self.assert_state_is([SafeClass, OtherSafeClass])
        self.assertEqual(getter(), [])

        with safe_globals([SafeClass]):
            with safe_globals([SafeClass]):
                self.assert_state_is([SafeClass])
            self.assertEqual(getter(), [])
        self.assertEqual(getter(), [])

        values = [SafeClass]
        context = safe_globals(values)
        context.__enter__()
        values[:] = [OtherSafeClass]
        context.__exit__(None, None, None)
        self.assert_state_is([SafeClass])

    def test_context_invalid_iterables_raise_on_enter_and_preserve_state(self):
        for value, message in (
            (None, "'NoneType' object is not iterable"),
            (1, "'int' object is not iterable"),
            (object(), "'object' object is not iterable"),
        ):
            with self.subTest(value=repr(value)):
                self.serialization.clear_safe_globals()
                self.serialization.add_safe_globals([SafeClass])
                context = self.serialization.safe_globals(value)
                self.assertEqual(context.__dict__, {"safe_globals": value})
                with self.assertRaises(TypeError) as raised:
                    context.__enter__()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assert_state_is([SafeClass])
                with self.assertRaises(TypeError) as exit_raised:
                    context.__exit__(None, None, None)
                self.assertEqual(str(exit_raised.exception), message)
                self.assertEqual(exit_raised.exception.args, (message,))
                self.assert_state_is([SafeClass])

        self.serialization.clear_safe_globals()
        self.serialization.add_safe_globals([SafeClass])
        context = self.serialization.safe_globals("xy")
        self.assertIsNone(context.__enter__())
        self.assert_state_is([SafeClass, "x", "y"])
        self.assertIsNone(context.__exit__(None, None, None))
        self.assert_state_is([SafeClass])

    def test_signature_metadata_documentation_import_copy_and_pickle(self):
        serialization = importlib.import_module("torch_rs.serialization")
        self.assertIs(torch.serialization, serialization)
        self.assertIs(sys.modules["torch_rs.serialization"], serialization)
        self.assertIsNone(serialization.__doc__)

        expected_exports = [
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
        ]
        self.assertEqual(serialization.__all__, expected_exports)

        for name in (
            "clear_safe_globals",
            "get_safe_globals",
            "add_safe_globals",
        ):
            with self.subTest(name=name):
                function = getattr(serialization, name)
                self.assertIs(type(function), types.FunctionType)
                self.assertEqual(function.__module__, "torch_rs.serialization")
                self.assertIs(inspect.getmodule(function), serialization)
                self.assertIsNone(function.__defaults__)
                self.assertIsNone(function.__kwdefaults__)
                self.assertEqual(function.__dict__, {})
                self.assertFalse(hasattr(function, "__text_signature__"))
                self.assertEqual(function.__name__, name)
                self.assertEqual(function.__qualname__, name)
                self.assertEqual(function.__code__.co_freevars, ())
                self.assertEqual(function.__code__.co_cellvars, ())

        self.assertEqual(
            str(inspect.signature(serialization.clear_safe_globals)),
            "() -> None",
        )
        self.assertEqual(
            serialization.clear_safe_globals.__annotations__,
            {"return": None},
        )
        self.assertEqual(
            typing.get_type_hints(serialization.clear_safe_globals),
            {"return": type(None)},
        )
        self.assertEqual(
            inspect.cleandoc(serialization.clear_safe_globals.__doc__),
            inspect.cleandoc(CLEAR_DOC),
        )

        self.assertEqual(
            str(inspect.signature(serialization.get_safe_globals)),
            "() -> list[collections.abc.Callable | tuple[collections.abc.Callable, str]]",
        )
        self.assertEqual(
            serialization.get_safe_globals.__annotations__,
            {"return": SAFE_GLOBALS_ANNOTATION},
        )
        self.assertEqual(
            typing.get_type_hints(serialization.get_safe_globals),
            {"return": SAFE_GLOBALS_ANNOTATION},
        )
        self.assertEqual(
            inspect.cleandoc(serialization.get_safe_globals.__doc__),
            inspect.cleandoc(GET_DOC),
        )

        self.assertEqual(
            str(inspect.signature(serialization.add_safe_globals)),
            "(safe_globals: list[collections.abc.Callable | "
            "tuple[collections.abc.Callable, str]]) -> None",
        )
        self.assertEqual(
            serialization.add_safe_globals.__annotations__,
            {"safe_globals": SAFE_GLOBALS_ANNOTATION, "return": None},
        )
        self.assertEqual(
            typing.get_type_hints(serialization.add_safe_globals),
            {"safe_globals": SAFE_GLOBALS_ANNOTATION, "return": type(None)},
        )
        self.assertEqual(
            inspect.cleandoc(serialization.add_safe_globals.__doc__),
            inspect.cleandoc(ADD_DOC),
        )

        context_class = serialization.safe_globals
        self.assertIs(type(context_class), type)
        self.assertEqual(context_class.__module__, "torch_rs.serialization")
        self.assertIs(inspect.getmodule(context_class), serialization)
        self.assertEqual(context_class.__name__, "safe_globals")
        self.assertEqual(context_class.__qualname__, "safe_globals")
        self.assertEqual(context_class.__annotations__, {})
        self.assertEqual(
            inspect.cleandoc(context_class.__doc__),
            inspect.cleandoc(CONTEXT_DOC),
        )
        self.assertEqual(
            str(inspect.signature(context_class)),
            "(safe_globals: list[collections.abc.Callable | tuple[collections.abc.Callable, str]])",
        )
        self.assertEqual(len(context_class.__bases__), 1)
        self.assertEqual(
            context_class.__bases__[0].__module__,
            "torch_rs._weights_only_unpickler",
        )
        self.assertEqual(context_class.__bases__[0].__name__, "_safe_globals")

        method_expectations = {
            "__init__": (
                "(self, safe_globals: list[collections.abc.Callable | "
                "tuple[collections.abc.Callable, str]])",
                {"safe_globals": SAFE_GLOBALS_ANNOTATION},
                {"safe_globals": SAFE_GLOBALS_ANNOTATION},
            ),
            "__enter__": ("(self)", {}, {}),
            "__exit__": ("(self, type, value, tb)", {}, {}),
        }
        for name, (signature, annotations, type_hints) in method_expectations.items():
            with self.subTest(method=name):
                method = getattr(context_class, name)
                self.assertEqual(str(inspect.signature(method)), signature)
                self.assertEqual(method.__module__, "torch_rs._weights_only_unpickler")
                self.assertEqual(method.__qualname__, f"_safe_globals.{name}")
                self.assertEqual(method.__annotations__, annotations)
                self.assertEqual(typing.get_type_hints(method), type_hints)
                self.assertIsNone(method.__defaults__)
                self.assertIsNone(method.__kwdefaults__)
                self.assertEqual(method.__dict__, {})
                self.assertFalse(hasattr(method, "__text_signature__"))

        direct_import = {}
        exec(
            "from torch_rs.serialization import "
            "clear_safe_globals, get_safe_globals, add_safe_globals, safe_globals",
            direct_import,
        )
        for name in (
            "clear_safe_globals",
            "get_safe_globals",
            "add_safe_globals",
            "safe_globals",
        ):
            self.assertIs(direct_import[name], getattr(serialization, name))

        namespace = {}
        exec("from torch_rs.serialization import *", namespace)
        self.assertEqual(
            {name for name in namespace if not name.startswith("__")},
            set(expected_exports),
        )
        for name in expected_exports:
            self.assertIs(namespace[name], getattr(serialization, name))

        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        for name in (
            "clear_safe_globals",
            "get_safe_globals",
            "add_safe_globals",
            "safe_globals",
        ):
            self.assertFalse(hasattr(torch, name))
            self.assertNotIn(name, torch.__all__)
            self.assertNotIn(name, top_level_namespace)

        for value in (
            serialization.clear_safe_globals,
            serialization.get_safe_globals,
            serialization.add_safe_globals,
            serialization.safe_globals,
        ):
            self.assertIs(copy.copy(value), value)
            self.assertIs(copy.deepcopy(value), value)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(value=value, protocol=protocol):
                    payload = pickle.dumps(value, protocol=protocol)
                    self.assertIn(b"torch_rs.serialization", payload)
                    self.assertIs(pickle.loads(payload), value)

    def test_argument_errors_match_pytorch_2_13(self):
        serialization = self.serialization
        cases = (
            (
                lambda: serialization.clear_safe_globals(None),
                "clear_safe_globals() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: serialization.clear_safe_globals(None, None),
                "clear_safe_globals() takes 0 positional arguments but 2 were given",
            ),
            (
                lambda: serialization.clear_safe_globals(enabled=True),
                "clear_safe_globals() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: serialization.get_safe_globals(None),
                "get_safe_globals() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: serialization.get_safe_globals(None, None),
                "get_safe_globals() takes 0 positional arguments but 2 were given",
            ),
            (
                lambda: serialization.get_safe_globals(enabled=True),
                "get_safe_globals() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: serialization.add_safe_globals(),
                "add_safe_globals() missing 1 required positional argument: 'safe_globals'",
            ),
            (
                lambda: serialization.add_safe_globals([], []),
                "add_safe_globals() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: serialization.add_safe_globals(enabled=True),
                "add_safe_globals() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: serialization.add_safe_globals([], safe_globals=[]),
                "add_safe_globals() got multiple values for argument 'safe_globals'",
            ),
            (
                lambda: serialization.safe_globals(),
                "_safe_globals.__init__() missing 1 required positional argument: 'safe_globals'",
            ),
            (
                lambda: serialization.safe_globals([], []),
                "_safe_globals.__init__() takes 2 positional arguments but 3 were given",
            ),
            (
                lambda: serialization.safe_globals(enabled=True),
                "_safe_globals.__init__() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: serialization.safe_globals([], safe_globals=[]),
                "_safe_globals.__init__() got multiple values for argument 'safe_globals'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                self.serialization.add_safe_globals([SafeClass])
                before = self.serialization.get_safe_globals()
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assert_state_is(before)

    def test_reload_and_reimport_preserve_state_and_rebind_public_objects(self):
        original_module = self.serialization
        original_getter = original_module.get_safe_globals
        original_adder = original_module.add_safe_globals
        original_clearer = original_module.clear_safe_globals
        original_context_class = original_module.safe_globals
        module_name = original_module.__name__

        original_adder([SafeClass])
        self.assertIs(importlib.reload(original_module), original_module)
        reloaded_getter = original_module.get_safe_globals
        reloaded_adder = original_module.add_safe_globals
        reloaded_clearer = original_module.clear_safe_globals
        reloaded_context_class = original_module.safe_globals

        self.assertIs(torch.serialization, original_module)
        for old, new in (
            (original_getter, reloaded_getter),
            (original_adder, reloaded_adder),
            (original_clearer, reloaded_clearer),
            (original_context_class, reloaded_context_class),
        ):
            self.assertIsNot(old, new)
        self.assert_state_is([SafeClass])
        reloaded_adder([OtherSafeClass])
        self.assertEqual(
            set(original_getter()),
            {SafeClass, OtherSafeClass},
        )

        with original_context_class([safe_function]):
            self.assertEqual(
                set(reloaded_getter()),
                {SafeClass, OtherSafeClass, safe_function},
            )
        self.assert_state_is([SafeClass, OtherSafeClass])

        try:
            self.assertIs(sys.modules.pop(module_name), original_module)
            replacement_module = importlib.import_module(module_name)
            self.assertIsNot(replacement_module, original_module)
            self.assertIs(sys.modules[module_name], replacement_module)
            self.assertIs(torch.serialization, replacement_module)
            self.assertEqual(
                set(replacement_module.get_safe_globals()),
                {SafeClass, OtherSafeClass},
            )

            replacement_module.add_safe_globals([safe_function])
            self.assertEqual(
                set(original_getter()),
                {SafeClass, OtherSafeClass, safe_function},
            )
            original_clearer()
            self.assertEqual(reloaded_getter(), [])
            self.assertEqual(replacement_module.get_safe_globals(), [])
        finally:
            sys.modules[module_name] = original_module
            torch.serialization = original_module

    def test_save_load_scanning_and_unpickling_remain_unsupported(self):
        serialization = self.serialization
        self.assertFalse(hasattr(torch, "save"))
        self.assertFalse(hasattr(torch, "load"))
        for name in ("save", "load", "get_unsafe_globals_in_checkpoint"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(serialization, name))
                self.assertNotIn(name, serialization.__all__)

    def test_importing_reloading_and_calling_does_not_import_pytorch(self):
        script = r"""
import importlib
import pickle
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

class SafeClass:
    pass

def safe_function():
    pass

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

serialization = torch.serialization
assert serialization.get_safe_globals() == []
assert serialization.clear_safe_globals() is None
assert serialization.add_safe_globals([SafeClass, safe_function]) is None
assert set(serialization.get_safe_globals()) == {SafeClass, safe_function}
snapshot = serialization.get_safe_globals()
snapshot.clear()
assert set(serialization.get_safe_globals()) == {SafeClass, safe_function}
context = serialization.safe_globals([len])
assert pickle.loads(pickle.dumps(serialization.safe_globals)) is serialization.safe_globals
rehydrated = pickle.loads(pickle.dumps(context))
assert type(rehydrated) is serialization.safe_globals
with context as value:
    assert value is None
    assert set(serialization.get_safe_globals()) == {SafeClass, safe_function, len}
assert set(serialization.get_safe_globals()) == {SafeClass, safe_function}
old_getter = serialization.get_safe_globals
assert importlib.reload(serialization) is serialization
assert set(old_getter()) == {SafeClass, safe_function}
del sys.modules["torch_rs.serialization"]
replacement = importlib.import_module("torch_rs.serialization")
assert replacement is not serialization
assert torch.serialization is replacement
assert set(replacement.get_safe_globals()) == {SafeClass, safe_function}
replacement.clear_safe_globals()
assert old_getter() == []
assert not hasattr(replacement, "save")
assert not hasattr(replacement, "load")
assert not hasattr(replacement, "get_unsafe_globals_in_checkpoint")
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
