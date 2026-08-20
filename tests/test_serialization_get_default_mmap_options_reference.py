import contextlib
import copy
import importlib
import inspect
import mmap
import pickle
import pickletools
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
class SerializationDefaultMmapOptionsReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "serialization mmap option differentials require pinned "
                "PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def supported_state_outcome(self, module):
        function = module.serialization.get_default_mmap_options

        def query_outcome():
            before = module.is_grad_enabled()
            result = function()
            after = module.is_grad_enabled()
            return (
                before,
                result,
                type(result).__module__,
                type(result).__qualname__,
                after,
            )

        states = [query_outcome()]
        with module.no_grad():
            states.append(query_outcome())
            with module.no_grad():
                states.append(query_outcome())
            states.append(query_outcome())
        states.append(query_outcome())

        worker_count = 8
        barrier = threading.Barrier(worker_count)
        worker_states = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = module.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    worker_states[index] = query_outcome()
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))

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
        return states, worker_states

    def pickle_shape(self, function, protocol):
        shape = []
        for opcode, argument, _ in pickletools.genops(
            pickle.dumps(function, protocol=protocol)
        ):
            if opcode.name == "FRAME":
                argument = "<frame length>"
            elif isinstance(argument, str):
                argument = argument.replace("torch_rs", "torch")
            shape.append((opcode.name, argument))
        return shape

    def test_supported_default_threaded_and_grad_states_match_pytorch_2_13(self):
        self.assertEqual(
            self.supported_state_outcome(torch),
            self.supported_state_outcome(reference_torch),
        )

        expected = getattr(mmap, "MAP_PRIVATE", None)
        self.assertIs(torch.serialization.get_default_mmap_options(), expected)
        self.assertIs(
            reference_torch.serialization.get_default_mmap_options(),
            expected,
        )

    def mutation_outcome(self, module):
        getter = module.serialization.get_default_mmap_options
        setter = module.serialization.set_default_mmap_options
        private = getattr(mmap, "MAP_PRIVATE", None)
        shared = getattr(mmap, "MAP_SHARED", None)

        if private is None or shared is None:
            return getter(), []

        original = getter()
        states = []

        class ExpectedError(Exception):
            pass

        try:
            update = setter(shared)
            states.append(
                (
                    "direct",
                    getter(),
                    type(update).__name__,
                    update.prev,
                )
            )
            setter(private)
            with setter(shared) as outer_value:
                states.append(("outer", outer_value, getter()))
                with setter(private) as inner_value:
                    states.append(("inner", inner_value, getter()))
                states.append(("after_inner", getter()))
            states.append(("after_outer", getter()))

            try:
                with setter(shared):
                    states.append(("exception_body", getter()))
                    raise ExpectedError("restore after exceptional exit")
            except ExpectedError as error:
                states.append(("exception", type(error).__name__, str(error)))
            states.append(("after_exception", getter()))

            keyword_update = setter(flags=shared)
            states.append(
                (
                    "keyword",
                    getter(),
                    type(keyword_update).__name__,
                    keyword_update.prev,
                )
            )
        finally:
            setter(original)

        return original, states

    def test_mutation_and_context_manager_behavior_match_pytorch_2_13(self):
        actual = self.mutation_outcome(torch)
        expected = self.mutation_outcome(reference_torch)

        self.assertEqual(actual, expected)
        private = getattr(mmap, "MAP_PRIVATE", None)
        shared = getattr(mmap, "MAP_SHARED", None)
        if private is not None and shared is not None:
            self.assertEqual(
                actual[1],
                [
                    ("direct", shared, "set_default_mmap_options", private),
                    ("outer", None, shared),
                    ("inner", None, private),
                    ("after_inner", shared),
                    ("after_outer", private),
                    ("exception_body", shared),
                    (
                        "exception",
                        "ExpectedError",
                        "restore after exceptional exit",
                    ),
                    ("after_exception", private),
                    ("keyword", shared, "set_default_mmap_options", private),
                ],
            )
        self.assertIs(torch.is_grad_enabled(), True)
        self.assertIs(reference_torch.is_grad_enabled(), True)

    def threaded_mutation_outcome(self, module):
        getter = module.serialization.get_default_mmap_options
        setter = module.serialization.set_default_mmap_options
        private = getattr(mmap, "MAP_PRIVATE", None)
        shared = getattr(mmap, "MAP_SHARED", None)

        if private is None or shared is None:
            return getter(), []

        def threaded_values():
            worker_count = 8
            barrier = threading.Barrier(worker_count)
            values = [None] * worker_count
            errors = []

            def worker(index):
                try:
                    barrier.wait(timeout=10)
                    values[index] = getter()
                except BaseException as error:
                    errors.append((type(error).__name__, str(error)))

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
            return values

        original = getter()
        try:
            setter(private)
            with setter(shared):
                inside = threaded_values()
            after = threaded_values()
        finally:
            setter(original)
        return original, inside, after

    def test_process_global_thread_visibility_is_explicit(self):
        actual = self.threaded_mutation_outcome(torch)
        expected = self.threaded_mutation_outcome(reference_torch)

        self.assertEqual(actual[0], expected[0])
        private = getattr(mmap, "MAP_PRIVATE", None)
        shared = getattr(mmap, "MAP_SHARED", None)
        if private is not None and shared is not None:
            self.assertEqual(actual[1], [shared] * 8)
            self.assertEqual(actual[2], [private] * 8)
            # PyTorch 2.13 stores this config in a ContextVar, so a fresh
            # thread sees the default. torch_rs intentionally keeps the
            # documented process-global behavior in its persistent state.
            self.assertEqual(expected[1], [private] * 8)
            self.assertEqual(expected[2], [private] * 8)

    def reload_outcome(self, module):
        private = getattr(mmap, "MAP_PRIVATE", None)
        shared = getattr(mmap, "MAP_SHARED", None)
        if private is None or shared is None or sys.platform == "win32":
            return module.serialization.get_default_mmap_options(), ()

        original_module = module.serialization
        original_getter = original_module.get_default_mmap_options
        original_setter = original_module.set_default_mmap_options
        module_name = original_module.__name__
        original = original_getter()
        replacement_module = None

        try:
            original_setter(shared)
            reloaded_module = importlib.reload(original_module)
            outcomes = [
                reloaded_module is original_module,
                module.serialization is original_module,
                original_getter(),
                reloaded_module.get_default_mmap_options(),
            ]

            popped_module = sys.modules.pop(module_name)
            replacement_module = importlib.import_module(module_name)
            outcomes.extend(
                (
                    popped_module is original_module,
                    replacement_module is not original_module,
                    sys.modules[module_name] is replacement_module,
                    module.serialization is replacement_module,
                    original_getter(),
                    replacement_module.get_default_mmap_options(),
                )
            )

            replacement_module.set_default_mmap_options(private)
            outcomes.extend(
                (
                    original_getter(),
                    replacement_module.get_default_mmap_options(),
                )
            )
            with original_setter(shared):
                outcomes.extend(
                    (
                        original_getter(),
                        replacement_module.get_default_mmap_options(),
                    )
                )
            outcomes.extend(
                (
                    original_getter(),
                    replacement_module.get_default_mmap_options(),
                )
            )
            return original, tuple(outcomes)
        finally:
            state_owner = replacement_module or original_module
            state_owner.set_default_mmap_options(original)
            sys.modules[module_name] = original_module
            module.serialization = original_module

    def test_reload_and_reimport_state_matches_pytorch_2_13(self):
        actual = self.reload_outcome(torch)
        expected = self.reload_outcome(reference_torch)

        self.assertEqual(actual, expected)
        private = getattr(mmap, "MAP_PRIVATE", None)
        shared = getattr(mmap, "MAP_SHARED", None)
        if private is not None and shared is not None and sys.platform != "win32":
            self.assertEqual(
                actual[1],
                (
                    True,
                    True,
                    shared,
                    shared,
                    True,
                    True,
                    True,
                    True,
                    shared,
                    shared,
                    private,
                    private,
                    shared,
                    shared,
                    private,
                    private,
                ),
            )

    def test_signature_annotations_documentation_and_identity_match(self):
        actual_module = importlib.import_module("torch_rs.serialization")
        expected_module = importlib.import_module("torch.serialization")
        actual = actual_module.get_default_mmap_options
        expected = expected_module.get_default_mmap_options

        self.assertIs(torch.serialization, actual_module)
        self.assertIs(reference_torch.serialization, expected_module)
        self.assertIsNone(actual_module.__doc__)
        self.assertEqual(actual_module.__doc__, expected_module.__doc__)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
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

    def test_setter_signature_documentation_and_identity_match(self):
        actual_module = importlib.import_module("torch_rs.serialization")
        expected_module = importlib.import_module("torch.serialization")
        actual = actual_module.set_default_mmap_options
        expected = expected_module.set_default_mmap_options

        self.assertIs(type(actual), type)
        self.assertIs(type(expected), type)
        self.assertEqual(str(inspect.signature(actual)), str(inspect.signature(expected)))
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(typing.get_type_hints(actual), typing.get_type_hints(expected))
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"),
            expected.__module__,
        )
        self.assertIs(inspect.getmodule(actual), actual_module)
        self.assertIs(inspect.getmodule(expected), expected_module)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )

        for method_name in ("__init__", "__enter__", "__exit__"):
            with self.subTest(method=method_name):
                actual_method = getattr(actual, method_name)
                expected_method = getattr(expected, method_name)
                self.assertIs(type(actual_method), types.FunctionType)
                self.assertIs(type(expected_method), types.FunctionType)
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
                self.assertEqual(actual_method.__doc__, expected_method.__doc__)
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
                    hasattr(actual_method, "__text_signature__"),
                    hasattr(expected_method, "__text_signature__"),
                )

    def test_imports_exports_copy_and_pickle_match_pytorch_2_13(self):
        actual_module = torch.serialization
        expected_module = reference_torch.serialization
        actual = actual_module.get_default_mmap_options
        expected = expected_module.get_default_mmap_options
        supported_names = (
            "get_crc32_options",
            "set_crc32_options",
            "get_default_mmap_options",
            "set_default_mmap_options",
        )

        self.assertEqual(
            actual_module.__all__,
            [name for name in expected_module.__all__ if name in supported_names],
        )

        actual_package_import = {}
        expected_package_import = {}
        exec("from torch_rs import serialization", actual_package_import)
        exec("from torch import serialization", expected_package_import)
        self.assertIs(actual_package_import["serialization"], actual_module)
        self.assertIs(expected_package_import["serialization"], expected_module)

        actual_direct_import = {}
        expected_direct_import = {}
        exec(
            "from torch_rs.serialization import "
            "get_default_mmap_options, set_default_mmap_options",
            actual_direct_import,
        )
        exec(
            "from torch.serialization import "
            "get_default_mmap_options, set_default_mmap_options",
            expected_direct_import,
        )
        self.assertIs(actual_direct_import["get_default_mmap_options"], actual)
        self.assertIs(expected_direct_import["get_default_mmap_options"], expected)
        self.assertIs(
            actual_direct_import["set_default_mmap_options"],
            actual_module.set_default_mmap_options,
        )
        self.assertIs(
            expected_direct_import["set_default_mmap_options"],
            expected_module.set_default_mmap_options,
        )

        for module, function in (
            (actual_module, actual),
            (expected_module, expected),
        ):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertIs(namespace["get_default_mmap_options"], function)
            self.assertIs(
                namespace["set_default_mmap_options"],
                module.set_default_mmap_options,
            )
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            self.assertIs(
                copy.copy(module.set_default_mmap_options),
                module.set_default_mmap_options,
            )
            self.assertIs(
                copy.deepcopy(module.set_default_mmap_options),
                module.set_default_mmap_options,
            )

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("get_default_mmap_options", namespace)
            self.assertNotIn("set_default_mmap_options", namespace)

        for actual_value, expected_value in (
            (actual, expected),
            (
                actual_module.set_default_mmap_options,
                expected_module.set_default_mmap_options,
            ),
        ):
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(
                    name=actual_value.__name__,
                    protocol=protocol,
                ):
                    self.assertIs(
                        pickle.loads(pickle.dumps(actual_value, protocol)),
                        actual_value,
                    )
                    self.assertIs(
                        pickle.loads(pickle.dumps(expected_value, protocol)),
                        expected_value,
                    )
                    self.assertEqual(
                        self.pickle_shape(actual_value, protocol),
                        self.pickle_shape(expected_value, protocol),
                    )

    def test_argument_errors_match_pytorch_2_13(self):
        actual_getter = torch.serialization.get_default_mmap_options
        expected_getter = reference_torch.serialization.get_default_mmap_options
        actual_setter = torch.serialization.set_default_mmap_options
        expected_setter = reference_torch.serialization.set_default_mmap_options
        cases = (
            (lambda: actual_getter(None), lambda: expected_getter(None)),
            (
                lambda: actual_getter(None, None),
                lambda: expected_getter(None, None),
            ),
            (
                lambda: actual_getter(enabled=True),
                lambda: expected_getter(enabled=True),
            ),
            (
                lambda: actual_getter(None, enabled=True),
                lambda: expected_getter(None, enabled=True),
            ),
            (lambda: actual_setter(), lambda: expected_setter()),
            (
                lambda: actual_setter(None, None),
                lambda: expected_setter(None, None),
            ),
            (
                lambda: actual_setter(enabled=True),
                lambda: expected_setter(enabled=True),
            ),
            (
                lambda: actual_setter(None, flags=None),
                lambda: expected_setter(None, flags=None),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                actual_before = actual_getter()
                expected_before = expected_getter()
                self.assert_error_matches(actual_call, expected_call)
                self.assertIs(actual_getter(), actual_before)
                self.assertIs(expected_getter(), expected_before)

        self.assertEqual(actual_getter(**{}), expected_getter(**{}))

        if sys.platform != "win32" and all(
            hasattr(mmap, name) for name in ("MAP_PRIVATE", "MAP_SHARED")
        ):
            for flags in (0, None, "private", mmap.MAP_PRIVATE | mmap.MAP_SHARED):
                with self.subTest(invalid_flags=flags):
                    actual_before = actual_getter()
                    expected_before = expected_getter()
                    self.assert_error_matches(
                        lambda: actual_setter(flags),
                        lambda: expected_setter(flags),
                    )
                    self.assertIs(actual_getter(), actual_before)
                    self.assertIs(expected_getter(), expected_before)

    def test_windows_platform_error_matches_and_precedes_validation(self):
        actual_module = torch.serialization
        expected_module = reference_torch.serialization
        actual_platform_flag = actual_module._IS_WINDOWS
        expected_platform_flag = expected_module.IS_WINDOWS
        actual_getter = actual_module.get_default_mmap_options
        expected_getter = expected_module.get_default_mmap_options

        try:
            actual_module._IS_WINDOWS = True
            expected_module.IS_WINDOWS = True
            for flags in (getattr(mmap, "MAP_PRIVATE", None), 0, None):
                with self.subTest(flags=flags):
                    actual_before = actual_getter()
                    expected_before = expected_getter()
                    self.assert_error_matches(
                        lambda: actual_module.set_default_mmap_options(flags),
                        lambda: expected_module.set_default_mmap_options(flags),
                    )
                    self.assertIs(actual_getter(), actual_before)
                    self.assertIs(expected_getter(), expected_before)
        finally:
            actual_module._IS_WINDOWS = actual_platform_flag
            expected_module.IS_WINDOWS = expected_platform_flag

    def test_save_and_load_remain_unsupported(self):
        actual_module = torch.serialization
        expected_module = reference_torch.serialization

        self.assertTrue(hasattr(actual_module, "set_default_mmap_options"))
        for name in ("save", "load"):
            with self.subTest(name=name):
                self.assertTrue(hasattr(expected_module, name))
                self.assertFalse(hasattr(actual_module, name))
                self.assertNotIn(name, actual_module.__all__)

        self.assertFalse(hasattr(torch, "get_default_mmap_options"))
        self.assertFalse(hasattr(reference_torch, "get_default_mmap_options"))
        self.assertNotIn("get_default_mmap_options", torch.__all__)
        self.assertFalse(hasattr(torch, "set_default_mmap_options"))
        self.assertFalse(hasattr(reference_torch, "set_default_mmap_options"))
        self.assertNotIn("set_default_mmap_options", torch.__all__)


if __name__ == "__main__":
    unittest.main()
