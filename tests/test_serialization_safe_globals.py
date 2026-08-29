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
from collections.abc import Callable

import torch_rs as torch


SAFE_GLOBAL = Callable | tuple[Callable, str]

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


class SerializationSafeGlobalsTests(unittest.TestCase):
    def setUp(self):
        self.serialization = torch.serialization
        self.original_safe_globals = self.serialization.get_safe_globals()
        self.serialization.clear_safe_globals()

    def tearDown(self):
        self.serialization.clear_safe_globals()
        self.serialization.add_safe_globals(self.original_safe_globals)

    def assert_safe_global_set(self, *expected):
        self.assertEqual(set(self.serialization.get_safe_globals()), set(expected))
        for value in expected:
            self.assertTrue(
                any(observed is value for observed in self.serialization.get_safe_globals())
            )

    def test_default_empty_getter_copy_and_clear_preserve_grad_mode(self):
        getter = self.serialization.get_safe_globals
        clear = self.serialization.clear_safe_globals
        self.assertEqual(getter.__code__.co_freevars, ())
        self.assertEqual(getter.__code__.co_cellvars, ())
        self.assertEqual(clear.__code__.co_freevars, ())
        self.assertEqual(clear.__code__.co_cellvars, ())

        class A:
            pass

        def assert_empty_preserves_grad_mode(expected_grad_state):
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)
            first = getter()
            second = getter()
            self.assertEqual(first, [])
            self.assertEqual(second, [])
            self.assertIsNot(first, second)
            first.append(A)
            self.assertEqual(getter(), [])
            self.assertIsNone(clear())
            self.assertEqual(getter(), [])
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)

        assert_empty_preserves_grad_mode(True)
        with torch.no_grad():
            assert_empty_preserves_grad_mode(False)
            with torch.no_grad():
                assert_empty_preserves_grad_mode(False)
            assert_empty_preserves_grad_mode(False)
        assert_empty_preserves_grad_mode(True)

    def test_add_accepts_runtime_iterables_and_preserves_identity(self):
        class A:
            pass

        class B:
            pass

        class C:
            pass

        def fn():
            pass

        self.assertIsNone(self.serialization.add_safe_globals([A, B, fn]))
        self.assert_safe_global_set(A, B, fn)

        observed = self.serialization.get_safe_globals()
        observed.append(C)
        self.assert_safe_global_set(A, B, fn)

        self.assertIsNone(self.serialization.add_safe_globals([A, A, B]))
        self.assertEqual(len(self.serialization.get_safe_globals()), 3)
        self.assert_safe_global_set(A, B, fn)

        self.serialization.clear_safe_globals()
        self.assertIsNone(self.serialization.add_safe_globals("abc"))
        self.assertEqual(set(self.serialization.get_safe_globals()), {"a", "b", "c"})

        self.serialization.clear_safe_globals()
        self.assertIsNone(self.serialization.add_safe_globals({"x": A}))
        self.assertEqual(self.serialization.get_safe_globals(), ["x"])

        self.serialization.clear_safe_globals()
        self.assertIsNone(self.serialization.add_safe_globals((A, B)))
        self.assert_safe_global_set(A, B)

        self.serialization.clear_safe_globals()
        self.assertIsNone(
            self.serialization.add_safe_globals(value for value in (A, B))
        )
        self.assert_safe_global_set(A, B)

    def test_duplicates_equal_objects_and_tuple_entries_use_set_semantics(self):
        class A:
            pass

        class Eq:
            def __init__(self, label):
                self.label = label

            def __hash__(self):
                return 1

            def __eq__(self, other):
                return isinstance(other, Eq)

        first = Eq("first")
        second = Eq("second")
        self.serialization.add_safe_globals([first, second])
        observed = self.serialization.get_safe_globals()
        self.assertEqual(observed, [first])
        self.assertIs(observed[0], first)

        self.serialization.add_safe_globals([second])
        observed = self.serialization.get_safe_globals()
        self.assertEqual(observed, [first])
        self.assertIs(observed[0], first)

        tuple_entries = (
            (A, "custom.A"),
            (A,),
            (A, "custom.A", "extra"),
            ("not callable", "custom.name"),
            (A, 123),
        )
        for value in tuple_entries:
            with self.subTest(value=value):
                self.serialization.clear_safe_globals()
                self.assertIsNone(self.serialization.add_safe_globals([value]))
                self.assertEqual(self.serialization.get_safe_globals(), [value])

    def test_context_nesting_restoration_and_exception_behavior(self):
        class A:
            pass

        class B:
            pass

        class C:
            pass

        context = self.serialization.safe_globals([A, B, A])
        self.assertEqual(context.__dict__, {"safe_globals": [A, B, A]})
        self.assertIsNone(context.__enter__())
        self.assert_safe_global_set(A, B)
        self.assertIsNone(context.__exit__(None, None, None))
        self.assertEqual(self.serialization.get_safe_globals(), [])

        with self.serialization.safe_globals([A]):
            self.assert_safe_global_set(A)
            with self.serialization.safe_globals([A, B]):
                self.assert_safe_global_set(A, B)
            self.assertEqual(self.serialization.get_safe_globals(), [])
        self.assertEqual(self.serialization.get_safe_globals(), [])

        self.serialization.add_safe_globals([A])
        with self.serialization.safe_globals([B]):
            self.assert_safe_global_set(A, B)
            self.serialization.add_safe_globals([C])
            self.assert_safe_global_set(A, B, C)
        self.assert_safe_global_set(A, C)

        self.serialization.clear_safe_globals()
        with self.assertRaises(RuntimeError):
            with self.serialization.safe_globals([A]):
                self.assert_safe_global_set(A)
                raise RuntimeError("boom")
        self.assertEqual(self.serialization.get_safe_globals(), [])

    def test_state_is_process_global_and_visible_across_threads(self):
        class A:
            pass

        class B:
            pass

        serialization = self.serialization
        worker_ready = threading.Event()
        read_updated = threading.Event()
        observations = []
        errors = []

        serialization.add_safe_globals([A])

        def observer():
            try:
                observations.append(set(serialization.get_safe_globals()) == {A})
                worker_ready.set()
                if not read_updated.wait(timeout=10):
                    raise RuntimeError("timed out waiting for the updated state")
                observations.append(set(serialization.get_safe_globals()) == {A, B})
                observations.append(serialization.clear_safe_globals())
                observations.append(serialization.get_safe_globals() == [])
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=observer)
        thread.start()
        self.assertTrue(worker_ready.wait(timeout=10))
        serialization.add_safe_globals([B])
        read_updated.set()
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(observations, [True, True, None, True])
        self.assertEqual(serialization.get_safe_globals(), [])

    def test_signature_annotations_documentation_and_module_identity(self):
        serialization = importlib.import_module("torch_rs.serialization")
        add = serialization.add_safe_globals
        getter = serialization.get_safe_globals
        clear = serialization.clear_safe_globals
        context_class = serialization.safe_globals

        self.assertIs(torch.serialization, serialization)
        self.assertIs(sys.modules["torch_rs.serialization"], serialization)
        self.assertIsNone(serialization.__doc__)
        for function in (add, getter, clear):
            self.assertIs(type(function), types.FunctionType)
            self.assertEqual(function.__module__, "torch_rs.serialization")
            self.assertIs(inspect.getmodule(function), serialization)
            self.assertIsNone(function.__defaults__)
            self.assertIsNone(function.__kwdefaults__)
            self.assertEqual(function.__dict__, {})
            self.assertFalse(hasattr(function, "__text_signature__"))

        self.assertEqual(
            str(inspect.signature(add)),
            "(safe_globals: list[collections.abc.Callable | tuple[collections.abc.Callable, str]]) -> None",
        )
        self.assertEqual(
            add.__annotations__,
            {"safe_globals": list[SAFE_GLOBAL], "return": None},
        )
        self.assertEqual(
            typing.get_type_hints(add),
            {"safe_globals": list[SAFE_GLOBAL], "return": type(None)},
        )
        self.assertEqual(inspect.cleandoc(add.__doc__), inspect.cleandoc(ADD_DOC))

        self.assertEqual(
            str(inspect.signature(getter)),
            "() -> list[collections.abc.Callable | tuple[collections.abc.Callable, str]]",
        )
        self.assertEqual(getter.__annotations__, {"return": list[SAFE_GLOBAL]})
        self.assertEqual(
            typing.get_type_hints(getter),
            {"return": list[SAFE_GLOBAL]},
        )
        self.assertEqual(inspect.cleandoc(getter.__doc__), inspect.cleandoc(GET_DOC))

        self.assertEqual(str(inspect.signature(clear)), "() -> None")
        self.assertEqual(clear.__annotations__, {"return": None})
        self.assertEqual(typing.get_type_hints(clear), {"return": type(None)})
        self.assertEqual(
            inspect.cleandoc(clear.__doc__),
            inspect.cleandoc(CLEAR_DOC),
        )

        self.assertIs(type(context_class), type)
        self.assertEqual(context_class.__module__, "torch_rs.serialization")
        self.assertEqual(context_class.__name__, "safe_globals")
        self.assertEqual(context_class.__qualname__, "safe_globals")
        self.assertIs(inspect.getmodule(context_class), serialization)
        self.assertEqual(context_class.__annotations__, {})
        self.assertEqual(
            str(inspect.signature(context_class)),
            "(safe_globals: list[collections.abc.Callable | tuple[collections.abc.Callable, str]])",
        )
        self.assertIn(
            "Context-manager that adds certain globals as safe",
            context_class.__doc__,
        )
        self.assertTrue(hasattr(context_class, "__text_signature__"))

    def test_imports_exports_copy_and_pickle_use_the_canonical_module(self):
        serialization = self.serialization
        exports = {
            "LoadEndianness": serialization.LoadEndianness,
            "get_crc32_options": serialization.get_crc32_options,
            "set_crc32_options": serialization.set_crc32_options,
            "get_default_load_endianness": (
                serialization.get_default_load_endianness
            ),
            "set_default_load_endianness": (
                serialization.set_default_load_endianness
            ),
            "get_default_mmap_options": serialization.get_default_mmap_options,
            "set_default_mmap_options": serialization.set_default_mmap_options,
            "clear_safe_globals": serialization.clear_safe_globals,
            "get_safe_globals": serialization.get_safe_globals,
            "add_safe_globals": serialization.add_safe_globals,
            "safe_globals": serialization.safe_globals,
        }

        self.assertEqual(serialization.__all__, list(exports))

        package_import = {}
        exec("from torch_rs import serialization", package_import)
        self.assertIs(package_import["serialization"], serialization)

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
            self.assertIs(direct_import[name], exports[name])

        serialization_namespace = {}
        exec("from torch_rs.serialization import *", serialization_namespace)
        self.assertEqual(
            {
                name
                for name in serialization_namespace
                if not name.startswith("__")
            },
            set(exports),
        )
        for name, value in exports.items():
            self.assertIs(serialization_namespace[name], value)

        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        for name in ("serialization", *exports):
            self.assertNotIn(name, torch.__all__)
            self.assertNotIn(name, top_level_namespace)

        for name, value in exports.items():
            self.assertIs(copy.copy(value), value)
            self.assertIs(copy.deepcopy(value), value)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(name=name, protocol=protocol):
                    payload = pickle.dumps(value, protocol=protocol)
                    self.assertIn(b"torch_rs.serialization", payload)
                    self.assertIs(pickle.loads(payload), value)

    def test_argument_errors_and_invalid_iterables_preserve_state(self):
        add = self.serialization.add_safe_globals
        getter = self.serialization.get_safe_globals
        clear = self.serialization.clear_safe_globals
        context_class = self.serialization.safe_globals

        class A:
            pass

        class B:
            pass

        self.serialization.add_safe_globals([A])
        cases = (
            (
                lambda: getter(None),
                "get_safe_globals() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: getter(enabled=True),
                "get_safe_globals() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: clear(None),
                "clear_safe_globals() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: clear(enabled=True),
                "clear_safe_globals() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: add(),
                "add_safe_globals() missing 1 required positional argument: 'safe_globals'",
            ),
            (
                lambda: add([], []),
                "add_safe_globals() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: add(enabled=True),
                "add_safe_globals() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: add([], safe_globals=[B]),
                "add_safe_globals() got multiple values for argument 'safe_globals'",
            ),
            (
                lambda: context_class(),
                "_safe_globals.__init__() missing 1 required positional argument: 'safe_globals'",
            ),
            (
                lambda: context_class([], []),
                "_safe_globals.__init__() takes 2 positional arguments but 3 were given",
            ),
            (
                lambda: context_class(enabled=True),
                "_safe_globals.__init__() got an unexpected keyword argument 'enabled'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                before = getter()
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertEqual(getter(), before)

        invalid_iterables = (
            (lambda: add(None), "'NoneType' object is not iterable"),
            (lambda: add(1), "'int' object is not iterable"),
            (
                lambda: add(object()),
                "'object' object is not iterable",
            ),
            (lambda: add([[B]]), "unhashable type: 'list'"),
            (lambda: add([{B}]), "unhashable type: 'set'"),
            (lambda: add([B, {"k": B}]), "unhashable type: 'dict'"),
            (
                lambda: context_class(None).__enter__(),
                "'NoneType' object is not iterable",
            ),
            (
                lambda: context_class([[B]]).__enter__(),
                "unhashable type: 'list'",
            ),
        )
        for call, message in invalid_iterables:
            with self.subTest(message=message):
                before = getter()
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertEqual(getter(), before)

        self.assertIsNone(add(safe_globals=[B]))
        self.assert_safe_global_set(A, B)

    def test_reload_and_reimport_preserve_process_state(self):
        serialization = self.serialization

        class A:
            pass

        class B:
            pass

        class C:
            pass

        original_getter = serialization.get_safe_globals
        original_adder = serialization.add_safe_globals
        original_clear = serialization.clear_safe_globals
        original_context_class = serialization.safe_globals
        module_name = serialization.__name__

        original_adder([A])
        self.assertIs(importlib.reload(serialization), serialization)
        self.assertIs(torch.serialization, serialization)
        self.assertIsNot(serialization.get_safe_globals, original_getter)
        self.assertIsNot(serialization.safe_globals, original_context_class)
        self.assert_safe_global_set(A)

        serialization.add_safe_globals([B])
        self.assertEqual(set(original_getter()), {A, B})
        self.assert_safe_global_set(A, B)

        try:
            self.assertIs(sys.modules.pop(module_name), serialization)
            replacement_module = importlib.import_module(module_name)

            self.assertIsNot(replacement_module, serialization)
            self.assertIs(sys.modules[module_name], replacement_module)
            self.assertIs(torch.serialization, replacement_module)
            self.assertEqual(set(original_getter()), {A, B})
            self.assertEqual(set(replacement_module.get_safe_globals()), {A, B})

            replacement_module.add_safe_globals([C])
            self.assertEqual(set(original_getter()), {A, B, C})
            self.assertEqual(
                set(replacement_module.get_safe_globals()),
                {A, B, C},
            )

            self.assertIsNone(original_clear())
            self.assertEqual(original_getter(), [])
            self.assertEqual(replacement_module.get_safe_globals(), [])
        finally:
            sys.modules[module_name] = serialization
            torch.serialization = serialization

    def test_save_load_and_scanning_remain_unsupported(self):
        serialization = self.serialization

        self.assertEqual(
            {name for name in vars(serialization) if not name.startswith("_")},
            {
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
            },
        )
        for name in ("save", "load", "get_unsafe_globals_in_checkpoint"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(serialization, name))
                self.assertNotIn(name, serialization.__all__)

        for name in (
            "clear_safe_globals",
            "get_safe_globals",
            "add_safe_globals",
            "safe_globals",
            "save",
            "load",
        ):
            with self.subTest(top_level_name=name):
                self.assertFalse(hasattr(torch, name))
                self.assertNotIn(name, torch.__all__)

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

class A:
    pass

class B:
    pass

serialization = torch.serialization
getter = serialization.get_safe_globals
adder = serialization.add_safe_globals
clear = serialization.clear_safe_globals
context_class = serialization.safe_globals
assert getter() == []
assert adder([A, B, A]) is None
assert set(getter()) == {A, B}
copy_of_state = getter()
copy_of_state.append(object())
assert set(getter()) == {A, B}
with context_class([A]):
    assert set(getter()) == {A, B}
assert getter() == [B]
with context_class([B]):
    assert getter() == [B]
    adder([A])
assert getter() == [A]
clear()
assert getter() == []
for value in (getter, adder, clear, context_class):
    assert copy.copy(value) is value
    assert copy.deepcopy(value) is value
    assert pickle.loads(pickle.dumps(value)) is value

assert importlib.reload(serialization) is serialization
assert serialization.get_safe_globals() == []
old_getter = serialization.get_safe_globals
old_adder = serialization.add_safe_globals
del sys.modules["torch_rs.serialization"]
replacement = importlib.import_module("torch_rs.serialization")
assert replacement is not serialization
assert torch.serialization is replacement
assert old_getter() == []
assert replacement.add_safe_globals([A]) is None
assert old_getter() == [A]
assert old_adder([B]) is None
assert set(replacement.get_safe_globals()) == {A, B}
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
