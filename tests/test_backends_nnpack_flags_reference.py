import copy
import importlib
import inspect
import pickle
import pickletools
import re
import sys
import threading
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class _RejectTruthiness:
    def __bool__(self):
        raise AssertionError("flags must not request truthiness")


class _ContextBodyError(Exception):
    pass


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class NnpackFlagsReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "backends.nnpack.flags differentials require pinned "
                "PyTorch 2.13.0"
            )

    def setUp(self):
        self.actual = importlib.import_module("torch_rs.backends.nnpack")
        self.expected = importlib.import_module("torch.backends.nnpack")
        self.original_actual = torch._C._get_nnpack_enabled()
        self.original_expected = reference_torch._C._get_nnpack_enabled()
        self.actual.set_flags(True)
        self.expected.set_flags(True)

    def tearDown(self):
        self.actual.set_flags(self.original_actual)
        self.expected.set_flags(self.original_expected)

    def normalize(self, value):
        value = str(value).replace("torch_rs", "torch")
        return re.sub(r"0x[0-9a-fA-F]+", "0x...", value)

    def capture_error(self, call):
        try:
            call()
        except Exception as error:
            return (
                type(error).__name__,
                self.normalize(error),
                tuple(self.normalize(argument) for argument in error.args),
            )
        self.fail("expected the call to fail")

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

    def behavior_contract(self, root, module):
        outcomes = []
        for initial in (False, True):
            module.set_flags(initial)
            context = module.flags()
            outcomes.append(root._C._get_nnpack_enabled())
            entered = context.__enter__()
            outcomes.append(
                (
                    entered,
                    root._C._get_nnpack_enabled(),
                    context.__exit__(None, None, None),
                    root._C._get_nnpack_enabled(),
                )
            )

            for enabled in (False, True):
                module.set_flags(initial)
                context = module.flags(enabled=enabled)
                state_before_entry = root._C._get_nnpack_enabled()
                entered = context.__enter__()
                state_inside = root._C._get_nnpack_enabled()
                exit_result = context.__exit__(None, None, None)
                outcomes.append(
                    (
                        state_before_entry,
                        entered,
                        state_inside,
                        exit_result,
                        root._C._get_nnpack_enabled(),
                    )
                )

        module.set_flags(True)
        with module.flags(False) as outer:
            outer_state = root._C._get_nnpack_enabled()
            with module.flags(True) as inner:
                inner_state = root._C._get_nnpack_enabled()
            after_inner = root._C._get_nnpack_enabled()
        outcomes.append(
            (
                outer,
                outer_state,
                inner,
                inner_state,
                after_inner,
                root._C._get_nnpack_enabled(),
            )
        )

        marker = _ContextBodyError("body failed")
        try:
            with module.flags(False) as entered:
                state_before_error = root._C._get_nnpack_enabled()
                raise marker
        except Exception as error:
            outcomes.append(
                (
                    entered,
                    state_before_error,
                    error is marker,
                    type(error).__name__,
                    error.args,
                    root._C._get_nnpack_enabled(),
                )
            )
        else:
            self.fail("the context suppressed its body exception")

        decorator_observations = []

        @module.flags(False)
        def decorated(value):
            decorator_observations.append(root._C._get_nnpack_enabled())
            return value

        outcomes.append(
            (
                decorated("first"),
                root._C._get_nnpack_enabled(),
                decorated("second"),
                root._C._get_nnpack_enabled(),
                decorator_observations,
            )
        )
        return outcomes

    def invalid_contract(self, root, module):
        outcomes = []
        invalid_values = (None, 0, 1, object(), _RejectTruthiness())
        for state in (False, True):
            for value in invalid_values:
                module.set_flags(state)
                context = module.flags(value)
                before_entry = root._C._get_nnpack_enabled()
                error = self.capture_error(context.__enter__)
                outcomes.append(
                    (before_entry, error, root._C._get_nnpack_enabled())
                )

        module.set_flags(True)
        for call in (
            lambda: module.flags(True, False),
            lambda: module.flags(_enabled=True),
            lambda: module.flags(False, enabled=True),
        ):
            outcomes.append(
                (
                    self.capture_error(call),
                    root._C._get_nnpack_enabled(),
                )
            )
        return outcomes

    def thread_contract(self, root, module):
        module.set_flags(True)
        worker_entered = threading.Event()
        main_context_exited = threading.Event()
        observations = []
        errors = []

        def worker():
            try:
                with module.flags(False) as entered:
                    observations.append(
                        ("worker-enter", entered, root._C._get_nnpack_enabled())
                    )
                    worker_entered.set()
                    if not main_context_exited.wait(timeout=10):
                        raise RuntimeError("timed out waiting for main context")
                    observations.append(
                        ("worker-resume", root._C._get_nnpack_enabled())
                    )
                observations.append(
                    ("worker-exit", root._C._get_nnpack_enabled())
                )
            except BaseException as error:
                errors.append((type(error).__name__, self.normalize(error)))
                worker_entered.set()

        thread = threading.Thread(target=worker)
        thread.start()
        worker_ready = worker_entered.wait(timeout=10)
        state_after_worker = root._C._get_nnpack_enabled()
        try:
            with module.flags(True) as entered:
                main_observation = (entered, root._C._get_nnpack_enabled())
            state_after_main = root._C._get_nnpack_enabled()
        finally:
            main_context_exited.set()
            thread.join(timeout=10)

        return (
            worker_ready,
            state_after_worker,
            main_observation,
            state_after_main,
            not thread.is_alive(),
            errors,
            observations,
            root._C._get_nnpack_enabled(),
        )

    def context_metadata_contract(self, module):
        context = module.flags(False)
        shallow = copy.copy(context)
        deep_error = self.capture_error(lambda: copy.deepcopy(context))
        pickle_errors = [
            self.capture_error(
                lambda protocol=protocol: pickle.dumps(context, protocol=protocol)
            )
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
        ]
        return (
            type(context).__module__,
            type(context).__qualname__,
            context.__doc__,
            context.func is module.flags.__wrapped__,
            context.args,
            context.kwds,
            shallow is not context,
            shallow.gen is context.gen,
            deep_error,
            pickle_errors,
        )

    def reload_contract(self, root, module):
        module.set_flags(True)
        old_flags = module.flags
        old_wrapped = old_flags.__wrapped__
        namespace = module.__dict__
        active_context = old_flags(False)
        entry_result = active_context.__enter__()
        state_before_reload = root._C._get_nnpack_enabled()
        reloaded = importlib.reload(module)
        state_after_reload = root._C._get_nnpack_enabled()
        exit_result = active_context.__exit__(None, None, None)
        state_after_exit = root._C._get_nnpack_enabled()

        context_results = []
        for function in (old_flags, module.flags):
            with function(False) as entered:
                context_results.append(
                    (entered, root._C._get_nnpack_enabled())
                )
            context_results.append(root._C._get_nnpack_enabled())

        old_pickle_error = self.capture_error(lambda: pickle.dumps(old_flags))
        return (
            entry_result,
            state_before_reload,
            reloaded is module,
            module.__dict__ is namespace,
            root.backends.nnpack is module,
            sys.modules[module.__name__] is module,
            module.flags is not old_flags,
            module.flags.__wrapped__ is not old_wrapped,
            state_after_reload,
            exit_result,
            state_after_exit,
            context_results,
            old_pickle_error,
            pickle.loads(pickle.dumps(module.flags)) is module.flags,
        )

    def test_context_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.behavior_contract(torch, self.actual),
            self.behavior_contract(reference_torch, self.expected),
        )

    def test_deferred_validation_and_binding_match_pytorch_2_13(self):
        self.assertEqual(
            self.invalid_contract(torch, self.actual),
            self.invalid_contract(reference_torch, self.expected),
        )

    def test_thread_visibility_matches_pytorch_2_13(self):
        self.assertEqual(
            self.thread_contract(torch, self.actual),
            self.thread_contract(reference_torch, self.expected),
        )

    def test_metadata_imports_copying_and_pickling_match_pytorch_2_13(self):
        actual = self.actual
        expected = self.expected
        actual_function = actual.flags
        expected_function = expected.flags
        actual_wrapped = actual_function.__wrapped__
        expected_wrapped = expected_function.__wrapped__

        self.assertIs(torch.backends.nnpack, actual)
        self.assertIs(reference_torch.backends.nnpack, expected)
        self.assertIs(type(actual), types.ModuleType)
        self.assertIs(type(expected), types.ModuleType)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(actual.__all__, expected.__all__)
        self.assertEqual(
            {name for name in vars(actual) if not name.startswith("_")},
            {
                name
                for name in vars(expected)
                if name
                in {
                    "contextmanager",
                    "flags",
                    "is_available",
                    "set_flags",
                    "torch",
                }
            },
        )

        for actual_item, expected_item in (
            (actual_function, expected_function),
            (actual_wrapped, expected_wrapped),
        ):
            self.assertIs(type(actual_item), types.FunctionType)
            self.assertIs(type(expected_item), types.FunctionType)
            self.assertEqual(
                str(inspect.signature(actual_item)),
                str(inspect.signature(expected_item)),
            )
            self.assertEqual(
                inspect.get_annotations(actual_item),
                inspect.get_annotations(expected_item),
            )
            self.assertEqual(actual_item.__name__, expected_item.__name__)
            self.assertEqual(actual_item.__qualname__, expected_item.__qualname__)
            self.assertEqual(
                actual_item.__module__.replace("torch_rs", "torch"),
                expected_item.__module__,
            )
            self.assertEqual(actual_item.__doc__, expected_item.__doc__)
            self.assertEqual(actual_item.__defaults__, expected_item.__defaults__)
            self.assertEqual(
                actual_item.__kwdefaults__,
                expected_item.__kwdefaults__,
            )
            self.assertEqual(
                set(actual_item.__dict__),
                set(expected_item.__dict__),
            )
            self.assertEqual(
                hasattr(actual_item, "__text_signature__"),
                hasattr(expected_item, "__text_signature__"),
            )
            self.assertEqual(
                actual_item.__code__.co_names,
                expected_item.__code__.co_names,
            )
            self.assertEqual(
                actual_item.__code__.co_freevars,
                expected_item.__code__.co_freevars,
            )
            self.assertEqual(
                actual_item.__code__.co_cellvars,
                expected_item.__code__.co_cellvars,
            )

        self.assertIs(inspect.getmodule(actual_function), actual)
        self.assertIs(inspect.getmodule(expected_function), expected)
        self.assertIs(inspect.getmodule(actual_wrapped), actual)
        self.assertIs(inspect.getmodule(expected_wrapped), expected)

        for package_name, module in (("torch_rs", actual), ("torch", expected)):
            backend_import = {}
            function_import = {}
            wildcard = {}
            exec(f"from {package_name}.backends import nnpack", backend_import)
            exec(
                f"from {package_name}.backends.nnpack import flags",
                function_import,
            )
            exec(f"from {package_name}.backends.nnpack import *", wildcard)
            self.assertIs(backend_import["nnpack"], module)
            self.assertIs(function_import["flags"], module.flags)
            self.assertIs(wildcard["flags"], module.flags)
            self.assertEqual(
                {name for name in wildcard if not name.startswith("__")},
                {"flags", "is_available", "set_flags"},
            )

        for function in (actual_function, expected_function):
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(actual_function, protocol)),
                    actual_function,
                )
                self.assertIs(
                    pickle.loads(pickle.dumps(expected_function, protocol)),
                    expected_function,
                )
                self.assertEqual(
                    self.pickle_shape(actual_function, protocol),
                    self.pickle_shape(expected_function, protocol),
                )

        self.assertEqual(
            self.context_metadata_contract(actual),
            self.context_metadata_contract(expected),
        )

    def test_reload_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_contract(torch, self.actual),
            self.reload_contract(reference_torch, self.expected),
        )

    def test_execution_remains_outside_the_supported_scope(self):
        self.assertTrue(hasattr(self.actual, "flags"))
        self.assertTrue(hasattr(self.expected, "flags"))
        self.assertIs(self.actual.is_available(), False)
        self.assertFalse(hasattr(torch, "_nnpack_spatial_convolution"))
        self.assertTrue(hasattr(reference_torch, "_nnpack_spatial_convolution"))


if __name__ == "__main__":
    unittest.main()
