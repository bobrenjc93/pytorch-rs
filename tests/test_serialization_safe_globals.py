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


CLEAR_DOC = """
    Clears the list of globals that are safe for ``weights_only`` load.
    """

GET_DOC = """
    Returns the list of user-added globals that are safe for ``weights_only`` load.
    """

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


class _EqualCallable:
    def __init__(self, name):
        self.name = name

    def __call__(self):
        return self.name

    def __hash__(self):
        return 17

    def __eq__(self, other):
        return isinstance(other, _EqualCallable)


class _OneShotIterable:
    def __init__(self, values):
        self.values = values
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        if self.iterations != 1:
            raise AssertionError("safe globals iterable was consumed more than once")
        return iter(self.values)


class _FailingIterable:
    def __iter__(self):
        yield len
        raise RuntimeError("safe globals iteration failed")


class _PickleTarget:
    pass


class SerializationSafeGlobalsTests(unittest.TestCase):
    def setUp(self):
        self.serialization = torch.serialization
        self.original_safe_globals = self.serialization.get_safe_globals()
        self.serialization.clear_safe_globals()

    def tearDown(self):
        self.serialization.clear_safe_globals()
        self.serialization.add_safe_globals(self.original_safe_globals)

    def test_add_deduplicates_by_equality_and_preserves_entry_identity(self):
        first = _EqualCallable("first")
        equal_but_distinct = _EqualCallable("second")
        first_tuple = tuple([len, "custom.len"])
        equal_tuple = tuple([len, "custom.len"])
        self.assertIsNot(first_tuple, equal_tuple)

        result = self.serialization.add_safe_globals(
            [first, first, equal_but_distinct, len, first_tuple, equal_tuple]
        )
        self.assertIsNone(result)

        snapshot = self.serialization.get_safe_globals()
        self.assertEqual(len(snapshot), 3)
        self.assertTrue(any(entry is first for entry in snapshot))
        self.assertFalse(any(entry is equal_but_distinct for entry in snapshot))
        self.assertTrue(any(entry is len for entry in snapshot))
        self.assertTrue(any(entry is first_tuple for entry in snapshot))
        self.assertFalse(any(entry is equal_tuple for entry in snapshot))

        self.serialization.add_safe_globals([equal_but_distinct, equal_tuple, str])
        updated = self.serialization.get_safe_globals()
        self.assertEqual(len(updated), 4)
        self.assertTrue(any(entry is first for entry in updated))
        self.assertFalse(any(entry is equal_but_distinct for entry in updated))
        self.assertTrue(any(entry is first_tuple for entry in updated))
        self.assertFalse(any(entry is equal_tuple for entry in updated))
        self.assertTrue(any(entry is str for entry in updated))

    def test_get_returns_fresh_list_snapshots_and_clear_replaces_state(self):
        first = object()
        second = object()
        self.serialization.add_safe_globals([first, second])

        snapshot = self.serialization.get_safe_globals()
        another_snapshot = self.serialization.get_safe_globals()
        self.assertIs(type(snapshot), list)
        self.assertIsNot(snapshot, another_snapshot)
        self.assertEqual(set(snapshot), {first, second})
        self.assertEqual(set(another_snapshot), {first, second})
        self.assertTrue(any(entry is first for entry in snapshot))
        self.assertTrue(any(entry is second for entry in snapshot))

        snapshot.clear()
        snapshot.append(object())
        self.assertEqual(set(self.serialization.get_safe_globals()), {first, second})

        before_clear = another_snapshot
        self.assertIsNone(self.serialization.clear_safe_globals())
        self.assertEqual(self.serialization.get_safe_globals(), [])
        self.assertEqual(set(before_clear), {first, second})
        self.assertIsNot(
            self.serialization.get_safe_globals(),
            self.serialization.get_safe_globals(),
        )

    def test_add_accepts_runtime_iterables_without_entry_validation(self):
        one_shot = _OneShotIterable((len, str, len))
        self.assertIsNone(self.serialization.add_safe_globals(one_shot))
        self.assertEqual(one_shot.iterations, 1)
        self.assertEqual(set(self.serialization.get_safe_globals()), {len, str})

        self.serialization.clear_safe_globals()
        self.assertIsNone(self.serialization.add_safe_globals(iter((int, float))))
        self.assertEqual(set(self.serialization.get_safe_globals()), {int, float})

        self.serialization.clear_safe_globals()
        self.assertIsNone(self.serialization.add_safe_globals("aba"))
        self.assertEqual(set(self.serialization.get_safe_globals()), {"a", "b"})

        self.serialization.clear_safe_globals()
        self.assertIsNone(self.serialization.add_safe_globals({"left": 1, "right": 2}))
        self.assertEqual(
            set(self.serialization.get_safe_globals()),
            {"left", "right"},
        )

        marker = object()
        self.serialization.clear_safe_globals()
        self.assertIsNone(self.serialization.add_safe_globals(safe_globals=[marker]))
        self.assertIs(self.serialization.get_safe_globals()[0], marker)

    def test_invalid_iterables_preserve_the_existing_registry(self):
        marker = object()
        self.serialization.add_safe_globals([marker])
        cases = (
            (None, "'NoneType' object is not iterable"),
            (3, "'int' object is not iterable"),
            (1.5, "'float' object is not iterable"),
            ([len, []], "unhashable type: 'list'"),
            ([(len, [])], "unhashable type: 'list'"),
        )
        for value, message in cases:
            with self.subTest(value=repr(value)):
                with self.assertRaises(TypeError) as raised:
                    self.serialization.add_safe_globals(value)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                current = self.serialization.get_safe_globals()
                self.assertEqual(len(current), 1)
                self.assertIs(current[0], marker)

        with self.assertRaisesRegex(
            RuntimeError,
            "^safe globals iteration failed$",
        ):
            self.serialization.add_safe_globals(_FailingIterable())
        current = self.serialization.get_safe_globals()
        self.assertEqual(len(current), 1)
        self.assertIs(current[0], marker)

    def test_state_is_process_global_across_threads(self):
        first = object()
        second = object()
        worker_observations = []
        worker_errors = []

        self.serialization.add_safe_globals([first])

        def worker():
            try:
                initial = self.serialization.get_safe_globals()
                self.serialization.add_safe_globals([second])
                updated = self.serialization.get_safe_globals()
                worker_observations.append(
                    (
                        any(entry is first for entry in initial),
                        any(entry is first for entry in updated),
                        any(entry is second for entry in updated),
                    )
                )
            except BaseException as error:
                worker_errors.append(error)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(worker_errors, [])
        self.assertEqual(worker_observations, [(True, True, True)])
        current = self.serialization.get_safe_globals()
        self.assertTrue(any(entry is first for entry in current))
        self.assertTrue(any(entry is second for entry in current))

    def test_signature_annotations_documentation_and_module_identity(self):
        serialization = importlib.import_module("torch_rs.serialization")
        functions = {
            "clear_safe_globals": (
                "() -> None",
                {"return": None},
                {"return": type(None)},
                CLEAR_DOC,
            ),
            "get_safe_globals": (
                "() -> list[collections.abc.Callable | "
                "tuple[collections.abc.Callable, str]]",
                {"return": list[Callable | tuple[Callable, str]]},
                {"return": list[Callable | tuple[Callable, str]]},
                GET_DOC,
            ),
            "add_safe_globals": (
                "(safe_globals: list[collections.abc.Callable | "
                "tuple[collections.abc.Callable, str]]) -> None",
                {
                    "safe_globals": list[Callable | tuple[Callable, str]],
                    "return": None,
                },
                {
                    "safe_globals": list[Callable | tuple[Callable, str]],
                    "return": type(None),
                },
                ADD_DOC,
            ),
        }

        self.assertIs(torch.serialization, serialization)
        self.assertIs(sys.modules["torch_rs.serialization"], serialization)
        for name, (signature, annotations, hints, doc) in functions.items():
            with self.subTest(name=name):
                function = getattr(serialization, name)
                self.assertIs(type(function), types.FunctionType)
                self.assertEqual(function.__module__, "torch_rs.serialization")
                self.assertIs(inspect.getmodule(function), serialization)
                self.assertIsNone(function.__defaults__)
                self.assertIsNone(function.__kwdefaults__)
                self.assertEqual(function.__dict__, {})
                self.assertFalse(hasattr(function, "__text_signature__"))
                self.assertEqual(function.__code__.co_freevars, ())
                self.assertEqual(function.__code__.co_cellvars, ())
                self.assertEqual(str(inspect.signature(function)), signature)
                self.assertEqual(function.__annotations__, annotations)
                self.assertEqual(typing.get_type_hints(function), hints)
                self.assertEqual(function.__name__, name)
                self.assertEqual(function.__qualname__, name)
                self.assertEqual(
                    inspect.cleandoc(function.__doc__), inspect.cleandoc(doc)
                )

    def test_imports_exports_copy_and_pickle_use_canonical_functions(self):
        serialization = self.serialization
        exported_names = [
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
        ]
        functions = {
            name: getattr(serialization, name)
            for name in (
                "clear_safe_globals",
                "get_safe_globals",
                "add_safe_globals",
            )
        }

        self.assertEqual(serialization.__all__, exported_names)
        self.assertEqual(
            {name for name in vars(serialization) if not name.startswith("_")},
            set(exported_names),
        )

        direct_import = {}
        exec(
            "from torch_rs.serialization import "
            "clear_safe_globals, get_safe_globals, add_safe_globals",
            direct_import,
        )
        wildcard_import = {}
        exec("from torch_rs.serialization import *", wildcard_import)
        top_level_import = {}
        exec("from torch_rs import *", top_level_import)
        for name, function in functions.items():
            self.assertIs(direct_import[name], function)
            self.assertIs(wildcard_import[name], function)
            self.assertNotIn(name, torch.__all__)
            self.assertNotIn(name, top_level_import)
            self.assertFalse(hasattr(torch, name))
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(name=name, protocol=protocol):
                    payload = pickle.dumps(function, protocol=protocol)
                    self.assertIn(b"torch_rs.serialization", payload)
                    self.assertIs(pickle.loads(payload), function)

        self.assertFalse(hasattr(serialization, "safe_globals"))
        self.assertNotIn("safe_globals", serialization.__all__)
        for name in ("save", "load"):
            self.assertFalse(hasattr(serialization, name))
            self.assertNotIn(name, serialization.__all__)

    def test_registry_snapshots_have_ordinary_copy_and_pickle_behavior(self):
        custom_path = (_PickleTarget, "custom.PickleTarget")
        self.serialization.add_safe_globals([_PickleTarget, custom_path])
        snapshot = self.serialization.get_safe_globals()

        shallow = copy.copy(snapshot)
        self.assertIsNot(shallow, snapshot)
        self.assertTrue(any(entry is _PickleTarget for entry in shallow))
        self.assertTrue(any(entry is custom_path for entry in shallow))

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                restored = pickle.loads(pickle.dumps(snapshot, protocol=protocol))
                self.assertEqual(set(restored), {_PickleTarget, custom_path})
                self.assertTrue(any(entry is _PickleTarget for entry in restored))

    def test_argument_errors_do_not_change_state(self):
        marker = object()
        self.serialization.add_safe_globals([marker])
        getter = self.serialization.get_safe_globals
        clearer = self.serialization.clear_safe_globals
        adder = self.serialization.add_safe_globals
        cases = (
            (
                lambda: getter(None),
                "get_safe_globals() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: getter(value=None),
                "get_safe_globals() got an unexpected keyword argument 'value'",
            ),
            (
                lambda: clearer(None),
                "clear_safe_globals() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: clearer(value=None),
                "clear_safe_globals() got an unexpected keyword argument 'value'",
            ),
            (
                lambda: adder(),
                "add_safe_globals() missing 1 required positional argument: "
                "'safe_globals'",
            ),
            (
                lambda: adder([], []),
                "add_safe_globals() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: adder(values=[]),
                "add_safe_globals() got an unexpected keyword argument 'values'",
            ),
            (
                lambda: adder([], safe_globals=[]),
                "add_safe_globals() got multiple values for argument 'safe_globals'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                current = getter()
                self.assertEqual(len(current), 1)
                self.assertIs(current[0], marker)

    def test_reload_and_reimport_share_registry_with_existing_functions(self):
        serialization = self.serialization
        old_add = serialization.add_safe_globals
        old_get = serialization.get_safe_globals
        old_clear = serialization.clear_safe_globals
        first = object()
        second = object()
        third = object()

        old_add([first])
        self.assertIs(importlib.reload(serialization), serialization)
        self.assertIs(torch.serialization, serialization)
        self.assertTrue(any(entry is first for entry in old_get()))
        self.assertTrue(
            any(entry is first for entry in serialization.get_safe_globals())
        )

        serialization.add_safe_globals([second])
        self.assertEqual(len(old_get()), 2)
        self.assertTrue(any(entry is second for entry in old_get()))

        module_name = serialization.__name__
        try:
            self.assertIs(sys.modules.pop(module_name), serialization)
            replacement = importlib.import_module(module_name)
            self.assertIsNot(replacement, serialization)
            self.assertIs(torch.serialization, replacement)
            self.assertTrue(
                any(entry is first for entry in replacement.get_safe_globals())
            )
            self.assertTrue(
                any(entry is second for entry in replacement.get_safe_globals())
            )

            replacement.add_safe_globals([third])
            self.assertTrue(any(entry is third for entry in old_get()))
            self.assertIsNone(old_clear())
            self.assertEqual(replacement.get_safe_globals(), [])
        finally:
            sys.modules[module_name] = serialization
            torch.serialization = serialization

    def test_importing_and_using_registry_does_not_import_pytorch(self):
        script = r"""
import importlib
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

serialization = torch.serialization
serialization.clear_safe_globals()
assert serialization.get_safe_globals() == []
assert serialization.add_safe_globals([len, (str, "custom.str"), len]) is None
snapshot = serialization.get_safe_globals()
assert len(snapshot) == 2
assert any(entry is len for entry in snapshot)
assert any(entry == (str, "custom.str") for entry in snapshot)
snapshot.clear()
assert len(serialization.get_safe_globals()) == 2
assert importlib.reload(serialization) is serialization
assert len(serialization.get_safe_globals()) == 2
assert serialization.clear_safe_globals() is None
assert serialization.get_safe_globals() == []
assert not hasattr(serialization, "safe_globals")
assert not hasattr(serialization, "load")
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
