import collections.abc
import copy
import importlib
import inspect
import pickle
import sys
import threading
import types
import typing
import unittest

import torch_rs as torch


def _first_safe_global():
    pass


def _second_safe_global():
    pass


class _EqualRegistrant:
    def __hash__(self):
        return 1

    def __eq__(self, other):
        return isinstance(other, _EqualRegistrant)


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


class SerializationSafeGlobalsTests(unittest.TestCase):
    def setUp(self):
        self.serialization = torch.serialization
        self.original_safe_globals = self.serialization.get_safe_globals()
        self.serialization.clear_safe_globals()

    def tearDown(self):
        self.serialization.clear_safe_globals()
        self.serialization.add_safe_globals(self.original_safe_globals)

    def test_add_consumes_once_deduplicates_and_preserves_tuple_entries(self):
        serialization = self.serialization
        tuple_entry = (_second_safe_global, "renamed.module.SecondGlobal")

        class OnePassIterable:
            def __init__(self):
                self.iterations = 0
                self.yielded = []

            def __iter__(self):
                self.iterations += 1
                if self.iterations != 1:
                    raise AssertionError("safe globals iterable was consumed twice")
                for value in (
                    _first_safe_global,
                    tuple_entry,
                    _first_safe_global,
                    tuple_entry,
                ):
                    self.yielded.append(value)
                    yield value

        values = OnePassIterable()
        self.assertIsNone(serialization.add_safe_globals(values))
        self.assertEqual(values.iterations, 1)
        self.assertEqual(len(values.yielded), 4)

        observed = serialization.get_safe_globals()
        self.assertIs(type(observed), list)
        self.assertEqual(set(observed), {_first_safe_global, tuple_entry})
        self.assertEqual(len(observed), 2)
        self.assertTrue(any(value is tuple_entry for value in observed))

        self.assertIsNone(
            serialization.add_safe_globals(
                safe_globals=(value for value in [_second_safe_global])
            )
        )
        self.assertEqual(
            set(serialization.get_safe_globals()),
            {_first_safe_global, _second_safe_global, tuple_entry},
        )

    def test_equal_entries_keep_the_existing_set_representative(self):
        first = _EqualRegistrant()
        second = _EqualRegistrant()

        self.assertIsNone(self.serialization.add_safe_globals([first, second]))
        observed = self.serialization.get_safe_globals()
        self.assertEqual(len(observed), 1)
        self.assertIs(observed[0], first)

        self.assertIsNone(self.serialization.add_safe_globals([second]))
        observed = self.serialization.get_safe_globals()
        self.assertEqual(len(observed), 1)
        self.assertIs(observed[0], first)

    def test_get_returns_fresh_lists_without_copying_entries(self):
        tuple_entry = (_second_safe_global, "renamed.module.SecondGlobal")
        self.serialization.add_safe_globals([_first_safe_global, tuple_entry])

        first = self.serialization.get_safe_globals()
        second = self.serialization.get_safe_globals()
        self.assertIsNot(first, second)
        self.assertEqual(set(first), set(second))
        self.assertTrue(any(value is tuple_entry for value in first))
        self.assertTrue(any(value is tuple_entry for value in second))

        first.clear()
        self.assertEqual(
            set(self.serialization.get_safe_globals()),
            {_first_safe_global, tuple_entry},
        )
        first.extend([_second_safe_global])
        self.assertNotIn(
            _second_safe_global,
            self.serialization.get_safe_globals(),
        )

    def test_clear_returns_none_and_replaces_all_entries(self):
        self.serialization.add_safe_globals(
            [_first_safe_global, (_second_safe_global, "renamed.Second")]
        )
        self.assertIsNone(self.serialization.clear_safe_globals())
        self.assertEqual(self.serialization.get_safe_globals(), [])
        self.assertIsNone(self.serialization.clear_safe_globals())
        self.assertEqual(self.serialization.get_safe_globals(), [])

    def test_hashability_and_iteration_failures_leave_state_unchanged(self):
        serialization = self.serialization
        serialization.add_safe_globals([_first_safe_global])

        class RaisingIterable:
            def __iter__(self):
                yield _second_safe_global
                raise RuntimeError("iteration failed")

        class RaisingHash:
            def __hash__(self):
                raise LookupError("hash failed")

        cases = (
            ([[]], TypeError, "unhashable type: 'list'"),
            (
                [(_second_safe_global, [])],
                TypeError,
                "unhashable type: 'list'",
            ),
            (1, TypeError, "'int' object is not iterable"),
            (None, TypeError, "'NoneType' object is not iterable"),
            (RaisingIterable(), RuntimeError, "iteration failed"),
            ([RaisingHash()], LookupError, "hash failed"),
        )
        for values, error_type, message in cases:
            with self.subTest(values=repr(values)):
                before = serialization.get_safe_globals()
                with self.assertRaises(error_type) as raised:
                    serialization.add_safe_globals(values)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertEqual(
                    set(serialization.get_safe_globals()),
                    set(before),
                )

    def test_state_is_process_global_and_visible_across_threads(self):
        serialization = self.serialization
        tuple_entry = (_second_safe_global, "renamed.module.SecondGlobal")
        worker_ready = threading.Event()
        read_update = threading.Event()
        observations = []
        errors = []

        serialization.add_safe_globals([_first_safe_global])

        def observer():
            try:
                observations.append(
                    set(serialization.get_safe_globals()) == {_first_safe_global}
                )
                worker_ready.set()
                if not read_update.wait(timeout=10):
                    raise RuntimeError("timed out waiting for registry update")
                observations.append(
                    set(serialization.get_safe_globals())
                    == {_first_safe_global, tuple_entry}
                )
                observations.append(serialization.clear_safe_globals())
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=observer)
        thread.start()
        self.assertTrue(worker_ready.wait(timeout=10))
        self.assertIsNone(serialization.add_safe_globals([tuple_entry]))
        read_update.set()
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(observations, [True, True, None])
        self.assertEqual(serialization.get_safe_globals(), [])

        worker_results = []

        def writer():
            worker_results.append(
                serialization.add_safe_globals([_second_safe_global])
            )
            worker_results.append(
                set(serialization.get_safe_globals()) == {_second_safe_global}
            )

        thread = threading.Thread(target=writer)
        thread.start()
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(worker_results, [None, True])
        self.assertEqual(
            set(serialization.get_safe_globals()),
            {_second_safe_global},
        )

    def test_signatures_annotations_documentation_and_module_identity(self):
        serialization = importlib.import_module("torch_rs.serialization")
        functions = {
            "clear_safe_globals": serialization.clear_safe_globals,
            "get_safe_globals": serialization.get_safe_globals,
            "add_safe_globals": serialization.add_safe_globals,
        }
        safe_global_type = collections.abc.Callable | tuple[
            collections.abc.Callable, str
        ]

        self.assertIs(torch.serialization, serialization)
        self.assertIs(sys.modules["torch_rs.serialization"], serialization)
        self.assertIsNone(serialization.__doc__)
        for name, function in functions.items():
            with self.subTest(name=name):
                self.assertIs(type(function), types.FunctionType)
                self.assertEqual(function.__module__, "torch_rs.serialization")
                self.assertIs(inspect.getmodule(function), serialization)
                self.assertEqual(function.__name__, name)
                self.assertEqual(function.__qualname__, name)
                self.assertIsNone(function.__defaults__)
                self.assertIsNone(function.__kwdefaults__)
                self.assertEqual(function.__dict__, {})
                self.assertEqual(function.__code__.co_freevars, ())
                self.assertEqual(function.__code__.co_cellvars, ())
                self.assertFalse(hasattr(function, "__text_signature__"))

        self.assertEqual(
            str(inspect.signature(functions["clear_safe_globals"])),
            "() -> None",
        )
        self.assertEqual(
            functions["clear_safe_globals"].__annotations__,
            {"return": None},
        )
        self.assertEqual(
            inspect.cleandoc(functions["clear_safe_globals"].__doc__),
            inspect.cleandoc(CLEAR_DOC),
        )

        self.assertEqual(
            str(inspect.signature(functions["get_safe_globals"])),
            "() -> list[collections.abc.Callable | "
            "tuple[collections.abc.Callable, str]]",
        )
        self.assertEqual(
            functions["get_safe_globals"].__annotations__,
            {"return": list[safe_global_type]},
        )
        self.assertEqual(
            inspect.cleandoc(functions["get_safe_globals"].__doc__),
            inspect.cleandoc(GET_DOC),
        )

        self.assertEqual(
            str(inspect.signature(functions["add_safe_globals"])),
            "(safe_globals: list[collections.abc.Callable | "
            "tuple[collections.abc.Callable, str]]) -> None",
        )
        self.assertEqual(
            functions["add_safe_globals"].__annotations__,
            {"safe_globals": list[safe_global_type], "return": None},
        )
        self.assertEqual(
            inspect.cleandoc(functions["add_safe_globals"].__doc__),
            inspect.cleandoc(ADD_DOC),
        )

        self.assertEqual(
            typing.get_type_hints(functions["clear_safe_globals"]),
            {"return": type(None)},
        )
        self.assertEqual(
            typing.get_type_hints(functions["get_safe_globals"]),
            {"return": list[safe_global_type]},
        )
        self.assertEqual(
            typing.get_type_hints(functions["add_safe_globals"]),
            {"safe_globals": list[safe_global_type], "return": type(None)},
        )

    def test_imports_exports_copy_and_pickle_use_the_canonical_module(self):
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
        self.assertEqual(serialization.__all__, exported_names)

        package_import = {}
        exec("from torch_rs import serialization", package_import)
        self.assertIs(package_import["serialization"], serialization)

        direct_import = {}
        exec(
            "from torch_rs.serialization import "
            "clear_safe_globals, get_safe_globals, add_safe_globals",
            direct_import,
        )
        for name in exported_names[-3:]:
            self.assertIs(direct_import[name], getattr(serialization, name))

        wildcard_import = {}
        exec("from torch_rs.serialization import *", wildcard_import)
        self.assertEqual(
            {name for name in wildcard_import if not name.startswith("__")},
            set(exported_names),
        )

        self.assertNotIn("serialization", torch.__all__)
        top_level_import = {}
        exec("from torch_rs import *", top_level_import)
        self.assertIs(torch.serialization, serialization)
        self.assertNotIn("serialization", top_level_import)
        for name in exported_names[-3:]:
            self.assertFalse(hasattr(torch, name))
            self.assertNotIn(name, torch.__all__)
            self.assertNotIn(name, top_level_import)

        for name in exported_names[-3:]:
            function = getattr(serialization, name)
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(name=name, protocol=protocol):
                    payload = pickle.dumps(function, protocol=protocol)
                    self.assertIn(b"torch_rs.serialization", payload)
                    self.assertIs(pickle.loads(payload), function)

    def test_argument_errors_do_not_change_the_registry(self):
        serialization = self.serialization
        serialization.add_safe_globals([_first_safe_global])
        cases = (
            (
                lambda: serialization.clear_safe_globals(None),
                "clear_safe_globals() takes 0 positional arguments but 1 was given",
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
                lambda: serialization.get_safe_globals(enabled=True),
                "get_safe_globals() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: serialization.add_safe_globals(),
                "add_safe_globals() missing 1 required positional argument: "
                "'safe_globals'",
            ),
            (
                lambda: serialization.add_safe_globals([], []),
                "add_safe_globals() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: serialization.add_safe_globals(globals=[]),
                "add_safe_globals() got an unexpected keyword argument 'globals'",
            ),
            (
                lambda: serialization.add_safe_globals([], safe_globals=[]),
                "add_safe_globals() got multiple values for argument 'safe_globals'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                before = serialization.get_safe_globals()
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertEqual(
                    set(serialization.get_safe_globals()),
                    set(before),
                )

    def test_reload_and_reimport_preserve_process_global_state(self):
        original_module = self.serialization
        old_add = original_module.add_safe_globals
        old_get = original_module.get_safe_globals
        old_clear = original_module.clear_safe_globals
        module_name = original_module.__name__
        tuple_entry = (_second_safe_global, "renamed.module.SecondGlobal")

        old_add([_first_safe_global])
        self.assertIs(importlib.reload(original_module), original_module)
        self.assertIs(torch.serialization, original_module)
        self.assertIsNot(original_module.add_safe_globals, old_add)
        self.assertIsNot(original_module.get_safe_globals, old_get)
        self.assertIsNot(original_module.clear_safe_globals, old_clear)
        self.assertEqual(set(old_get()), {_first_safe_global})
        self.assertEqual(
            set(original_module.get_safe_globals()),
            {_first_safe_global},
        )

        old_add([tuple_entry])
        self.assertEqual(
            set(original_module.get_safe_globals()),
            {_first_safe_global, tuple_entry},
        )

        try:
            self.assertIs(sys.modules.pop(module_name), original_module)
            replacement_module = importlib.import_module(module_name)
            self.assertIsNot(replacement_module, original_module)
            self.assertIs(sys.modules[module_name], replacement_module)
            self.assertIs(torch.serialization, replacement_module)
            self.assertEqual(
                set(replacement_module.get_safe_globals()),
                {_first_safe_global, tuple_entry},
            )

            self.assertIsNone(
                replacement_module.add_safe_globals([_second_safe_global])
            )
            self.assertEqual(
                set(old_get()),
                {_first_safe_global, _second_safe_global, tuple_entry},
            )
            self.assertIsNone(old_clear())
            self.assertEqual(replacement_module.get_safe_globals(), [])
        finally:
            sys.modules[module_name] = original_module
            torch.serialization = original_module

    def test_context_manager_save_and_load_remain_unsupported(self):
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
            },
        )
        for name in ("safe_globals", "save", "load"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(serialization, name))
                self.assertNotIn(name, serialization.__all__)


if __name__ == "__main__":
    unittest.main()
