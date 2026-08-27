import contextlib
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

import numpy as np

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class _BodyError(Exception):
    pass


class _RejectTruthiness:
    def __bool__(self):
        raise AssertionError("flags must not request truthiness")


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
        return re.sub(
            r"0x[0-9a-fA-F]+",
            "0x...",
            str(value).replace("torch_rs", "torch"),
        )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(
            self.normalize(actual_raised.exception),
            self.normalize(expected_raised.exception),
        )
        self.assertEqual(
            tuple(self.normalize(value) for value in actual_raised.exception.args),
            tuple(self.normalize(value) for value in expected_raised.exception.args),
        )

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

    def context_contract(self, root, module):
        module.set_flags(True)
        outcomes = []
        with module.flags() as entered:
            outcomes.append((entered, root._C._get_nnpack_enabled()))
            with module.flags(True) as nested_entered:
                outcomes.append(
                    (nested_entered, root._C._get_nnpack_enabled())
                )
            outcomes.append(root._C._get_nnpack_enabled())
        outcomes.append(root._C._get_nnpack_enabled())

        module.set_flags(False)
        try:
            with module.flags(enabled=True) as entered:
                outcomes.append((entered, root._C._get_nnpack_enabled()))
                raise _BodyError("body failed")
        except _BodyError as error:
            outcomes.append((type(error).__name__, str(error)))
        outcomes.append(root._C._get_nnpack_enabled())
        return outcomes

    def invalid_context_contract(self, root, module, values):
        outcomes = []
        for state in (False, True):
            for value in values:
                module.set_flags(state)
                context = module.flags(value)
                before_entry = root._C._get_nnpack_enabled()
                try:
                    context.__enter__()
                except Exception as error:
                    outcome = (
                        type(context) is contextlib._GeneratorContextManager,
                        before_entry,
                        type(error).__name__,
                        str(error),
                        error.args,
                        root._C._get_nnpack_enabled(),
                    )
                else:
                    self.fail("an invalid NNPACK flag entered successfully")
                outcomes.append(outcome)
        return outcomes

    def thread_contract(self, root, module):
        module.set_flags(True)
        entered = threading.Event()
        leave = threading.Event()
        observations = []
        errors = []

        def worker():
            try:
                with module.flags(False) as value:
                    observations.append((value, root._C._get_nnpack_enabled()))
                    entered.set()
                    if not leave.wait(timeout=10):
                        raise RuntimeError("timed out waiting to leave context")
                observations.append(root._C._get_nnpack_enabled())
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))
                entered.set()

        thread = threading.Thread(target=worker)
        thread.start()
        worker_ready = entered.wait(timeout=10)
        main_state = root._C._get_nnpack_enabled()
        with module.flags(True) as main_entered:
            nested_main_state = root._C._get_nnpack_enabled()
        restored_worker_state = root._C._get_nnpack_enabled()
        leave.set()
        thread.join(timeout=10)
        return (
            worker_ready,
            main_state,
            main_entered,
            nested_main_state,
            restored_worker_state,
            not thread.is_alive(),
            errors,
            observations,
            root._C._get_nnpack_enabled(),
        )

    def reload_contract(self, root, module):
        old_flags = module.flags
        active = old_flags(False)
        pending = old_flags(False)
        active_entered = active.__enter__()
        state_before_reload = root._C._get_nnpack_enabled()
        namespace = module.__dict__
        reloaded = importlib.reload(module)
        state_after_reload = root._C._get_nnpack_enabled()

        with module.flags(True) as nested_entered:
            nested_state = root._C._get_nnpack_enabled()
        state_after_nested = root._C._get_nnpack_enabled()
        active_exit = active.__exit__(None, None, None)
        state_after_active = root._C._get_nnpack_enabled()
        pending_entered = pending.__enter__()
        pending_state = root._C._get_nnpack_enabled()
        pending_exit = pending.__exit__(None, None, None)
        state_after_pending = root._C._get_nnpack_enabled()

        with old_flags(False) as old_entered:
            old_state = root._C._get_nnpack_enabled()
        state_after_old = root._C._get_nnpack_enabled()

        try:
            pickle.dumps(old_flags)
        except Exception as error:
            stale_pickle_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-fA-F]+", "0x...", str(error)).replace(
                    "torch_rs", "torch"
                ),
            )
        else:
            self.fail("a stale NNPACK flags function remained pickleable")

        return (
            active_entered,
            state_before_reload,
            reloaded is module,
            module.__dict__ is namespace,
            root.backends.nnpack is module,
            sys.modules[module.__name__] is module,
            module.flags is not old_flags,
            state_after_reload,
            nested_entered,
            nested_state,
            state_after_nested,
            active_exit,
            state_after_active,
            pending_entered,
            pending_state,
            pending_exit,
            state_after_pending,
            old_entered,
            old_state,
            state_after_old,
            stale_pickle_error,
            pickle.loads(pickle.dumps(module.flags)) is module.flags,
        )

    def test_context_entry_nesting_exceptions_and_threads_match_pytorch_2_13(self):
        self.assertEqual(
            self.context_contract(torch, self.actual),
            self.context_contract(reference_torch, self.expected),
        )
        self.assertEqual(
            self.thread_contract(torch, self.actual),
            self.thread_contract(reference_torch, self.expected),
        )

    def test_deferred_strict_boolean_validation_matches_pytorch_2_13(self):
        actual_values = (
            None,
            0,
            1,
            0.0,
            np.bool_(True),
            "",
            [],
            object(),
            _RejectTruthiness(),
            torch.tensor(True),
            torch.float32,
            torch.device("cpu"),
            torch.strided,
            torch.Size([1]),
            torch.finfo(torch.float32),
        )
        expected_values = (
            None,
            0,
            1,
            0.0,
            np.bool_(True),
            "",
            [],
            object(),
            _RejectTruthiness(),
            reference_torch.tensor(True),
            reference_torch.float32,
            reference_torch.device("cpu"),
            reference_torch.strided,
            reference_torch.Size([1]),
            reference_torch.finfo(reference_torch.float32),
        )
        self.assertEqual(
            self.invalid_context_contract(torch, self.actual, actual_values),
            self.invalid_context_contract(
                reference_torch,
                self.expected,
                expected_values,
            ),
        )

        cases = (
            (
                lambda: self.actual.flags(True, False),
                lambda: self.expected.flags(True, False),
            ),
            (
                lambda: self.actual.flags(_enabled=True),
                lambda: self.expected.flags(_enabled=True),
            ),
            (
                lambda: self.actual.flags(True, enabled=False),
                lambda: self.expected.flags(True, enabled=False),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.actual.set_flags(True)
                self.expected.set_flags(True)
                self.assert_error_matches(actual_call, expected_call)
                self.assertIs(torch._C._get_nnpack_enabled(), True)
                self.assertIs(reference_torch._C._get_nnpack_enabled(), True)

    def test_metadata_imports_copying_and_pickling_match_pytorch_2_13(self):
        actual = self.actual
        expected = self.expected
        actual_function = actual.flags
        expected_function = expected.flags
        actual_wrapped = actual_function.__wrapped__
        expected_wrapped = expected_function.__wrapped__

        self.assertEqual(actual.__all__, expected.__all__)
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        supported_names = {
            "contextmanager",
            "flags",
            "is_available",
            "set_flags",
            "torch",
        }
        self.assertEqual(
            {name for name in vars(actual) if not name.startswith("_")},
            {name for name in vars(expected) if name in supported_names},
        )
        self.assertIs(actual.contextmanager, contextlib.contextmanager)
        self.assertIs(expected.contextmanager, contextlib.contextmanager)
        self.assertIs(
            getattr(actual, "__allow_nonbracketed_mutation"),
            getattr(torch.backends, "__allow_nonbracketed_mutation"),
        )
        self.assertIs(
            getattr(expected, "__allow_nonbracketed_mutation"),
            getattr(reference_torch.backends, "__allow_nonbracketed_mutation"),
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
            self.assertEqual(
                inspect.isgeneratorfunction(actual_item),
                inspect.isgeneratorfunction(expected_item),
            )

        for package_name, module in (
            ("torch_rs", actual),
            ("torch", expected),
        ):
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
                {
                    name
                    for name in wildcard
                    if name in {"flags", "is_available", "set_flags"}
                },
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

        self.assert_error_matches(
            lambda: pickle.dumps(actual_wrapped),
            lambda: pickle.dumps(expected_wrapped),
        )
        self.assert_error_matches(
            lambda: pickle.dumps(actual_function()),
            lambda: pickle.dumps(expected_function()),
        )
        self.assertIs(type(actual_function()), type(expected_function()))

    def test_reload_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_contract(torch, self.actual),
            self.reload_contract(reference_torch, self.expected),
        )

    def test_nnpack_execution_remains_unsupported(self):
        self.assertFalse(hasattr(torch, "_nnpack_spatial_convolution"))
        self.assertTrue(hasattr(reference_torch, "_nnpack_spatial_convolution"))
        with self.actual.flags(True):
            self.assertIs(self.actual.is_available(), False)


if __name__ == "__main__":
    unittest.main()
