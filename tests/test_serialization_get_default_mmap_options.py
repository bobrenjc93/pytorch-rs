import asyncio
import contextlib
import copy
import importlib
import inspect
import mmap
import pickle
import subprocess
import sys
import threading
import types
import typing
import unittest

import torch_rs as torch


GETTER_DOC = """
    Get default mmap options for :func:`torch.load` with ``mmap=True``.

    Defaults to ``mmap.MAP_PRIVATE``.


    Returns:
        default_mmap_options: int
    """

SETTER_DOC = """
    Context manager or function to set default mmap options for :func:`torch.load` with ``mmap=True`` to flags.

    For now, only either ``mmap.MAP_PRIVATE`` or ``mmap.MAP_SHARED`` are supported.
    Please open an issue if you need any other option to be added here.

    .. note::
        This feature is currently not supported for Windows.

    Args:
        flags: ``mmap.MAP_PRIVATE`` or ``mmap.MAP_SHARED``
    """


class SerializationDefaultMmapOptionsTests(unittest.TestCase):
    def test_platform_default_is_exact_and_preserves_grad_mode(self):
        function = torch.serialization.get_default_mmap_options
        expected = getattr(mmap, "MAP_PRIVATE", None)

        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

        def assert_query_preserves_grad_mode(expected_grad_state):
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)
            self.assertIs(function(), expected)
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)

        assert_query_preserves_grad_mode(True)
        with torch.no_grad():
            assert_query_preserves_grad_mode(False)
            with torch.no_grad():
                assert_query_preserves_grad_mode(False)
            assert_query_preserves_grad_mode(False)
        assert_query_preserves_grad_mode(True)

        if expected is None:
            self.assertIsNone(function())
        else:
            self.assertIs(type(function()), int)
            self.assertEqual(function(), mmap.MAP_PRIVATE)

    def test_default_is_stable_across_threads_and_grad_modes(self):
        function = torch.serialization.get_default_mmap_options
        expected = getattr(mmap, "MAP_PRIVATE", None)
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = torch.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    results[index] = (
                        torch.is_grad_enabled(),
                        function() is expected,
                        torch.is_grad_enabled(),
                        function() is expected,
                        torch.is_grad_enabled(),
                    )
            except BaseException as error:
                errors.append(error)

        threads = [
            threading.Thread(target=worker, args=(index,))
            for index in range(worker_count)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        for index, result in enumerate(results):
            expected_grad_state = index % 2 == 0
            self.assertEqual(
                result,
                (
                    expected_grad_state,
                    True,
                    expected_grad_state,
                    True,
                    expected_grad_state,
                ),
            )

    def test_signature_annotations_documentation_and_module_identity(self):
        serialization = importlib.import_module("torch_rs.serialization")
        getter = serialization.get_default_mmap_options
        setter = serialization.set_default_mmap_options

        self.assertIs(torch.serialization, serialization)
        self.assertIs(sys.modules["torch_rs.serialization"], serialization)
        self.assertIsNone(serialization.__doc__)
        self.assertIs(type(getter), types.FunctionType)
        self.assertEqual(getter.__module__, "torch_rs.serialization")
        self.assertIs(inspect.getmodule(getter), serialization)
        self.assertIsNone(getter.__defaults__)
        self.assertIsNone(getter.__kwdefaults__)
        self.assertEqual(getter.__dict__, {})
        self.assertFalse(hasattr(getter, "__text_signature__"))
        self.assertEqual(str(inspect.signature(getter)), "() -> int | None")
        self.assertEqual(getter.__annotations__, {"return": int | None})
        self.assertEqual(typing.get_type_hints(getter), {"return": int | None})
        self.assertEqual(getter.__name__, "get_default_mmap_options")
        self.assertEqual(getter.__qualname__, "get_default_mmap_options")
        self.assertEqual(
            inspect.cleandoc(getter.__doc__),
            inspect.cleandoc(GETTER_DOC),
        )

        self.assertIs(type(setter), type)
        self.assertEqual(setter.__module__, "torch_rs.serialization")
        self.assertIs(inspect.getmodule(setter), serialization)
        self.assertEqual(str(inspect.signature(setter)), "(flags: int) -> None")
        self.assertEqual(setter.__annotations__, {})
        self.assertEqual(typing.get_type_hints(setter), {})
        self.assertEqual(setter.__name__, "set_default_mmap_options")
        self.assertEqual(setter.__qualname__, "set_default_mmap_options")
        self.assertEqual(
            inspect.cleandoc(setter.__doc__),
            inspect.cleandoc(SETTER_DOC),
        )
        self.assertEqual(
            str(inspect.signature(setter.__init__)),
            "(self, flags: int) -> None",
        )
        self.assertEqual(
            setter.__init__.__annotations__,
            {"flags": int, "return": None},
        )
        self.assertEqual(str(inspect.signature(setter.__enter__)), "(self) -> None")
        self.assertEqual(setter.__enter__.__annotations__, {"return": None})
        self.assertEqual(
            str(inspect.signature(setter.__exit__)),
            "(self, exc_type: Any, exc_value: Any, traceback: Any) -> None",
        )
        self.assertEqual(
            setter.__exit__.__annotations__,
            {
                "exc_type": typing.Any,
                "exc_value": typing.Any,
                "traceback": typing.Any,
                "return": None,
            },
        )

    def test_imports_exports_copy_and_pickle_use_the_canonical_module(self):
        serialization = torch.serialization
        getter = serialization.get_default_mmap_options
        setter = serialization.set_default_mmap_options
        exported_names = [
            "get_crc32_options",
            "set_crc32_options",
            "get_default_mmap_options",
            "set_default_mmap_options",
        ]

        self.assertEqual(serialization.__all__, exported_names)

        package_import = {}
        exec("from torch_rs import serialization", package_import)
        self.assertIs(package_import["serialization"], serialization)

        direct_import = {}
        exec(
            "from torch_rs.serialization import "
            "get_default_mmap_options, set_default_mmap_options",
            direct_import,
        )
        self.assertIs(direct_import["get_default_mmap_options"], getter)
        self.assertIs(direct_import["set_default_mmap_options"], setter)

        serialization_namespace = {}
        exec("from torch_rs.serialization import *", serialization_namespace)
        self.assertEqual(
            {
                name
                for name in serialization_namespace
                if not name.startswith("__")
            },
            set(exported_names),
        )
        self.assertIs(serialization_namespace["get_default_mmap_options"], getter)
        self.assertIs(serialization_namespace["set_default_mmap_options"], setter)

        self.assertNotIn("serialization", torch.__all__)
        self.assertNotIn("get_default_mmap_options", torch.__all__)
        self.assertNotIn("set_default_mmap_options", torch.__all__)
        self.assertFalse(hasattr(torch, "get_default_mmap_options"))
        self.assertFalse(hasattr(torch, "set_default_mmap_options"))
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("serialization", top_level_namespace)
        self.assertNotIn("get_default_mmap_options", top_level_namespace)
        self.assertNotIn("set_default_mmap_options", top_level_namespace)

        for value in (getter, setter):
            self.assertIs(copy.copy(value), value)
            self.assertIs(copy.deepcopy(value), value)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(name=value.__name__, protocol=protocol):
                    payload = pickle.dumps(value, protocol=protocol)
                    self.assertIn(b"torch_rs.serialization", payload)
                    self.assertIs(pickle.loads(payload), value)

    def test_argument_errors_match_pytorch_2_13(self):
        getter = torch.serialization.get_default_mmap_options
        setter = torch.serialization.set_default_mmap_options
        cases = (
            (
                lambda: getter(None),
                "get_default_mmap_options() takes 0 positional arguments but 1 "
                "was given",
            ),
            (
                lambda: getter(None, None),
                "get_default_mmap_options() takes 0 positional arguments but 2 "
                "were given",
            ),
            (
                lambda: getter(enabled=True),
                "get_default_mmap_options() got an unexpected keyword argument "
                "'enabled'",
            ),
            (
                lambda: getter(None, enabled=True),
                "get_default_mmap_options() got an unexpected keyword argument "
                "'enabled'",
            ),
            (
                lambda: setter(),
                "set_default_mmap_options.__init__() missing 1 required positional "
                "argument: 'flags'",
            ),
            (
                lambda: setter(None, None),
                "set_default_mmap_options.__init__() takes 2 positional arguments "
                "but 3 were given",
            ),
            (
                lambda: setter(enabled=True),
                "set_default_mmap_options.__init__() got an unexpected keyword "
                "argument 'enabled'",
            ),
            (
                lambda: setter(None, flags=None),
                "set_default_mmap_options.__init__() got multiple values for "
                "argument 'flags'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                before = getter()
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertIs(getter(), before)

        self.assertIs(getter(**{}), getattr(mmap, "MAP_PRIVATE", None))

    def test_invalid_flags_raise_without_changing_state(self):
        if sys.platform == "win32" or not all(
            hasattr(mmap, name) for name in ("MAP_PRIVATE", "MAP_SHARED")
        ):
            self.skipTest("mmap option mutation is unavailable")

        getter = torch.serialization.get_default_mmap_options
        setter = torch.serialization.set_default_mmap_options
        for flags in (0, None, "private", mmap.MAP_PRIVATE | mmap.MAP_SHARED):
            with self.subTest(flags=flags):
                before = getter()
                message = (
                    "Invalid argument in function set_default_mmap_options, "
                    "expected mmap.MAP_PRIVATE or mmap.MAP_SHARED, "
                    f"but got {flags}"
                )
                with self.assertRaises(ValueError) as raised:
                    setter(flags)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertIs(getter(), before)

    def test_immediate_updates_nested_contexts_and_exceptions_restore_state(self):
        if sys.platform == "win32" or not all(
            hasattr(mmap, name) for name in ("MAP_PRIVATE", "MAP_SHARED")
        ):
            self.skipTest("mmap option mutation is unavailable")

        getter = torch.serialization.get_default_mmap_options
        setter = torch.serialization.set_default_mmap_options
        private = mmap.MAP_PRIVATE
        shared = mmap.MAP_SHARED
        original = getter()

        class ExpectedError(Exception):
            pass

        try:
            update = setter(shared)
            self.assertIs(type(update), setter)
            self.assertIs(update.prev, original)
            self.assertIs(getter(), shared)

            setter(private)
            self.assertIs(getter(), private)
            with setter(shared) as outer_value:
                self.assertIsNone(outer_value)
                self.assertIs(getter(), shared)
                with setter(private) as inner_value:
                    self.assertIsNone(inner_value)
                    self.assertIs(getter(), private)
                self.assertIs(getter(), shared)
            self.assertIs(getter(), private)

            with self.assertRaises(ExpectedError):
                with setter(shared):
                    self.assertIs(getter(), shared)
                    raise ExpectedError("restore after exceptional exit")
            self.assertIs(getter(), private)

            keyword_update = setter(flags=shared)
            self.assertIs(type(keyword_update), setter)
            self.assertIs(getter(), shared)
        finally:
            setter(original)

    def test_overlapping_thread_contexts_are_isolated(self):
        if sys.platform == "win32" or not all(
            hasattr(mmap, name) for name in ("MAP_PRIVATE", "MAP_SHARED")
        ):
            self.skipTest("mmap option mutation is unavailable")

        getter = torch.serialization.get_default_mmap_options
        setter = torch.serialization.set_default_mmap_options
        private = mmap.MAP_PRIVATE
        shared = mmap.MAP_SHARED
        original = getter()
        first_entered = threading.Event()
        second_entered = threading.Event()
        first_exited = threading.Event()
        observations = {}
        errors = []

        def first_worker():
            try:
                with setter(shared):
                    observations["first_enter"] = getter()
                    first_entered.set()
                    if not second_entered.wait(timeout=10):
                        raise TimeoutError("second worker did not enter")
                    observations["first_before_exit"] = getter()
                observations["first_after_exit"] = getter()
            except BaseException as error:
                errors.append(error)
            finally:
                first_entered.set()
                first_exited.set()

        def second_worker():
            try:
                if not first_entered.wait(timeout=10):
                    raise TimeoutError("first worker did not enter")
                context = setter(private)
                observations["second_previous"] = context.prev
                with context:
                    observations["second_enter"] = getter()
                    second_entered.set()
                    if not first_exited.wait(timeout=10):
                        raise TimeoutError("first worker did not exit")
                    observations["second_after_first_exit"] = getter()
                observations["second_after_exit"] = getter()
            except BaseException as error:
                errors.append(error)
            finally:
                second_entered.set()

        try:
            setter(private)
            threads = [
                threading.Thread(target=first_worker),
                threading.Thread(target=second_worker),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)

            self.assertFalse(any(thread.is_alive() for thread in threads))
            self.assertEqual(errors, [])
            self.assertEqual(
                observations,
                {
                    "first_enter": shared,
                    "first_before_exit": shared,
                    "first_after_exit": private,
                    "second_previous": private,
                    "second_enter": private,
                    "second_after_first_exit": private,
                    "second_after_exit": private,
                },
            )
            self.assertIs(getter(), private)
        finally:
            setter(original)

    def test_overlapping_async_task_contexts_are_isolated(self):
        if sys.platform == "win32" or not all(
            hasattr(mmap, name) for name in ("MAP_PRIVATE", "MAP_SHARED")
        ):
            self.skipTest("mmap option mutation is unavailable")

        getter = torch.serialization.get_default_mmap_options
        setter = torch.serialization.set_default_mmap_options
        private = mmap.MAP_PRIVATE
        shared = mmap.MAP_SHARED
        original = getter()

        async def scenario():
            first_entered = asyncio.Event()
            second_entered = asyncio.Event()
            first_exited = asyncio.Event()
            observations = {}

            async def first_task():
                with setter(shared):
                    observations["first_enter"] = getter()
                    first_entered.set()
                    await asyncio.wait_for(second_entered.wait(), timeout=10)
                    observations["first_before_exit"] = getter()
                observations["first_after_exit"] = getter()
                first_exited.set()

            async def second_task():
                await asyncio.wait_for(first_entered.wait(), timeout=10)
                context = setter(private)
                observations["second_previous"] = context.prev
                with context:
                    observations["second_enter"] = getter()
                    second_entered.set()
                    await asyncio.wait_for(first_exited.wait(), timeout=10)
                    observations["second_after_first_exit"] = getter()
                observations["second_after_exit"] = getter()

            first = asyncio.create_task(first_task())
            second = asyncio.create_task(second_task())
            await asyncio.gather(first, second)
            return observations, getter()

        try:
            setter(private)
            observations, final_value = asyncio.run(scenario())
            self.assertEqual(
                observations,
                {
                    "first_enter": shared,
                    "first_before_exit": shared,
                    "first_after_exit": private,
                    "second_previous": private,
                    "second_enter": private,
                    "second_after_first_exit": private,
                    "second_after_exit": private,
                },
            )
            self.assertIs(final_value, private)
            self.assertIs(getter(), private)
        finally:
            setter(original)

    def test_reload_and_reimport_preserve_and_share_context_state(self):
        if sys.platform == "win32" or not all(
            hasattr(mmap, name) for name in ("MAP_PRIVATE", "MAP_SHARED")
        ):
            self.skipTest("mmap option mutation is unavailable")

        original_module = torch.serialization
        original_getter = original_module.get_default_mmap_options
        original_setter = original_module.set_default_mmap_options
        module_name = original_module.__name__
        original = original_getter()
        replacement_module = None

        try:
            original_setter(mmap.MAP_SHARED)
            self.assertIs(importlib.reload(original_module), original_module)
            self.assertIs(torch.serialization, original_module)
            self.assertIs(original_getter(), mmap.MAP_SHARED)
            self.assertIs(
                original_module.get_default_mmap_options(),
                mmap.MAP_SHARED,
            )

            self.assertIs(sys.modules.pop(module_name), original_module)
            replacement_module = importlib.import_module(module_name)

            self.assertIsNot(replacement_module, original_module)
            self.assertIs(sys.modules[module_name], replacement_module)
            self.assertIs(torch.serialization, replacement_module)
            self.assertIs(original_getter(), mmap.MAP_SHARED)
            self.assertIs(
                replacement_module.get_default_mmap_options(),
                mmap.MAP_SHARED,
            )

            replacement_module.set_default_mmap_options(mmap.MAP_PRIVATE)
            self.assertIs(original_getter(), mmap.MAP_PRIVATE)
            with original_setter(mmap.MAP_SHARED):
                self.assertIs(
                    replacement_module.get_default_mmap_options(),
                    mmap.MAP_SHARED,
                )
            self.assertIs(
                replacement_module.get_default_mmap_options(),
                mmap.MAP_PRIVATE,
            )
        finally:
            state_owner = replacement_module or original_module
            state_owner.set_default_mmap_options(original)
            sys.modules[module_name] = original_module
            torch.serialization = original_module

    def test_save_and_load_remain_unsupported(self):
        serialization = torch.serialization

        self.assertTrue(hasattr(serialization, "get_default_mmap_options"))
        self.assertTrue(hasattr(serialization, "set_default_mmap_options"))
        for name in ("save", "load"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(serialization, name))
                self.assertNotIn(name, serialization.__all__)

    def test_windows_platform_check_precedes_flag_validation(self):
        serialization = torch.serialization
        getter = serialization.get_default_mmap_options
        setter = serialization.set_default_mmap_options
        original_platform_flag = serialization._IS_WINDOWS
        before = getter()
        message = (
            "Changing the default mmap options is currently not supported for Windows"
        )

        try:
            serialization._IS_WINDOWS = True
            for flags in (getattr(mmap, "MAP_PRIVATE", None), 0, None):
                with self.subTest(flags=flags):
                    with self.assertRaises(RuntimeError) as raised:
                        setter(flags)
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))
                    self.assertIs(getter(), before)
        finally:
            serialization._IS_WINDOWS = original_platform_flag

    def test_unavailable_mmap_constants_are_not_valid_flags(self):
        if not all(hasattr(mmap, name) for name in ("MAP_PRIVATE", "MAP_SHARED")):
            self.skipTest("mmap constants are unavailable on this platform")

        script = r"""
import importlib
import mmap
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

missing_name = sys.argv[1]
available_name = (
    "MAP_SHARED" if missing_name == "MAP_PRIVATE" else "MAP_PRIVATE"
)
delattr(mmap, missing_name)
sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

serialization = torch.serialization
initial = getattr(mmap, "MAP_PRIVATE", None)
assert serialization.get_default_mmap_options() == initial
assert importlib.reload(serialization) is serialization
assert serialization.get_default_mmap_options() == initial

message = (
    "Invalid argument in function set_default_mmap_options, "
    "expected mmap.MAP_PRIVATE or mmap.MAP_SHARED, but got None"
)
try:
    serialization.set_default_mmap_options(None)
except ValueError as error:
    assert str(error) == message
    assert error.args == (message,)
else:
    raise AssertionError(f"None was accepted without {missing_name}")
assert serialization.get_default_mmap_options() == initial

available = getattr(mmap, available_name)
with serialization.set_default_mmap_options(available) as value:
    assert value is None
    assert serialization.get_default_mmap_options() == available
assert serialization.get_default_mmap_options() == initial
assert "set_default_mmap_options" in serialization.__all__
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
"""
        for missing_name in ("MAP_PRIVATE", "MAP_SHARED"):
            with self.subTest(missing_name=missing_name):
                completed = subprocess.run(
                    [sys.executable, "-c", script, missing_name],
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
