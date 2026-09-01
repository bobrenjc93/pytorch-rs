import contextlib
import copy
import decimal
import fractions
import importlib
import inspect
import numbers
import pickle
import pickletools
import re
import threading
import types
import typing
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class _ContextBodyError(Exception):
    pass


class _BoolProbe:
    def __init__(self, name, log, result=True, error=None):
        self.name = name
        self.log = log
        self.result = result
        self.error = error

    def __bool__(self):
        self.log.append(self.name)
        if self.error is not None:
            raise self.error
        return self.result


class _LenOnly:
    def __len__(self):
        return 1


class _FakeNumber(numbers.Number):
    pass


class _RegisteredNumber:
    pass


numbers.Number.register(_RegisteredNumber)


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class JitOptimizedExecutionReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "jit.optimized_execution differentials require pinned PyTorch 2.13.0"
            )

    def normalize(self, text):
        text = text.replace("torch_rs", "torch")
        return re.sub(r"0x[0-9a-fA-F]+", "0x...", text)

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(
            self.normalize(str(actual_raised.exception)),
            self.normalize(str(expected_raised.exception)),
        )
        self.assertEqual(
            tuple(self.normalize(str(arg)) for arg in actual_raised.exception.args),
            tuple(self.normalize(str(arg)) for arg in expected_raised.exception.args),
        )

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

    def context_outcome(self, module):
        states = [module.is_grad_enabled()]
        with module.jit.optimized_execution(True) as first_entered:
            states.append(module.is_grad_enabled())
            with module.jit.optimized_execution(False) as nested_entered:
                states.append(module.is_grad_enabled())
            states.append(module.is_grad_enabled())
        states.append(module.is_grad_enabled())

        with module.no_grad():
            states.append(module.is_grad_enabled())
            with module.jit.optimized_execution(False) as disabled_entered:
                states.append(module.is_grad_enabled())
                with module.jit.optimized_execution(True):
                    states.append(module.is_grad_enabled())
            states.append(module.is_grad_enabled())
        states.append(module.is_grad_enabled())

        error = _ContextBodyError("body failed")
        try:
            with module.jit.optimized_execution(True):
                raise error
        except _ContextBodyError as raised:
            propagated = raised is error
        else:
            propagated = False

        manual_context = module.jit.optimized_execution(True)
        manual_entered = manual_context.__enter__()
        manual_exited = manual_context.__exit__(None, None, None)

        return (
            first_entered,
            nested_entered,
            disabled_entered,
            states,
            manual_entered,
            manual_exited,
            propagated,
        )

    def validation_outcome(self, module):
        accepted = (
            True,
            False,
            None,
            0,
            1,
            2,
            -1,
            0.0,
            1.0,
            0j,
            1 + 0j,
            decimal.Decimal("0"),
            decimal.Decimal("1"),
            fractions.Fraction(0, 1),
            fractions.Fraction(1, 2),
            np.bool_(False),
            np.int64(1),
            np.array(False),
            np.array([True]),
            module.tensor(0.0),
            module.tensor([1.0]),
        )
        accepted_outcomes = []
        for value in accepted:
            context = module.jit.optimized_execution(value)
            accepted_outcomes.append(
                (
                    type(context).__module__,
                    type(context).__qualname__,
                    context.__enter__(),
                    context.__exit__(None, None, None),
                )
            )

        log = []
        for result in (False, True):
            context = module.jit.optimized_execution(
                _BoolProbe(str(result), log, result=result)
            )
            context.__enter__()
            context.__exit__(None, None, None)

        return accepted_outcomes, log

    def threaded_grad_outcome(self, module):
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        outcomes = [None] * worker_count
        errors = []

        def worker(index):
            try:
                grad_context = (
                    module.no_grad() if index % 2 else contextlib.nullcontext()
                )
                with grad_context:
                    barrier.wait(timeout=10)
                    before = module.is_grad_enabled()
                    with module.jit.optimized_execution(index % 3 == 0) as entered:
                        inside = module.is_grad_enabled()
                    outcomes[index] = (
                        before,
                        inside,
                        module.is_grad_enabled(),
                        entered,
                    )
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

        return (
            outcomes,
            sorted(errors),
            [thread.is_alive() for thread in threads],
            module.is_grad_enabled(),
        )

    def decorator_outcome(self, module):
        observations = []

        @module.jit.optimized_execution(False)
        def decorated(value: int, scale: int = 2, *, label: str = "x") -> tuple:
            """decorated doc"""
            observations.append((module.is_grad_enabled(), value, scale, label))
            if label == "raise":
                raise _ContextBodyError("decorated body failed")
            return value * scale, label

        first = decorated(3, label="first")
        with module.no_grad():
            second = decorated(4, 5)
        try:
            decorated(1, label="raise")
        except _ContextBodyError as error:
            raised = str(error)
        else:
            raised = None

        return (
            first,
            second,
            raised,
            observations,
            decorated.__name__,
            decorated.__qualname__.split(".")[-1],
            decorated.__doc__,
            inspect.signature(decorated),
            inspect.signature(decorated.__wrapped__),
            typing.get_type_hints(decorated),
        )

    def test_context_validation_decorator_and_grad_state_match(self):
        self.assertEqual(
            self.validation_outcome(torch),
            self.validation_outcome(reference_torch),
        )
        self.assertEqual(
            self.context_outcome(torch),
            self.context_outcome(reference_torch),
        )
        self.assertEqual(
            self.threaded_grad_outcome(torch),
            self.threaded_grad_outcome(reference_torch),
        )
        self.assertEqual(
            self.decorator_outcome(torch),
            self.decorator_outcome(reference_torch),
        )

    def test_invalid_values_and_binding_errors_match(self):
        invalid = (
            "",
            "x",
            [],
            [1],
            {},
            {1: 2},
            object(),
            _LenOnly(),
            _FakeNumber(),
            _RegisteredNumber(),
            np.array([]),
            np.array([True, False]),
        )
        for value in invalid:
            with self.subTest(value_type=type(value).__name__):
                self.assert_error_matches(
                    lambda value=value: torch.jit.optimized_execution(value).__enter__(),
                    lambda value=value: reference_torch.jit.optimized_execution(
                        value
                    ).__enter__(),
                )

        for call in (
            lambda function: function(),
            lambda function: function(True, False),
            lambda function: function(enabled=True),
            lambda function: function(True, should_optimize=False),
        ):
            with self.subTest(call=call):
                self.assert_error_matches(
                    lambda call=call: call(torch.jit.optimized_execution),
                    lambda call=call: call(reference_torch.jit.optimized_execution),
                )

    def test_metadata_imports_copy_pickle_and_reload_match(self):
        actual_jit = importlib.import_module("torch_rs.jit")
        expected_jit = importlib.import_module("torch.jit")
        actual_fuser = importlib.import_module("torch_rs.jit._fuser")
        expected_fuser = importlib.import_module("torch.jit._fuser")
        actual = actual_jit.optimized_execution
        expected = expected_jit.optimized_execution

        self.assertIs(torch.jit, actual_jit)
        self.assertIs(reference_torch.jit, expected_jit)
        self.assertIs(actual, actual_fuser.optimized_execution)
        self.assertIs(expected, expected_fuser.optimized_execution)
        self.assertEqual(type(actual), type(expected))
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertEqual(
            str(inspect.signature(actual)),
            str(inspect.signature(expected)),
        )
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(typing.get_type_hints(actual), typing.get_type_hints(expected))
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertFalse(hasattr(actual, "__text_signature__"))
        self.assertFalse(hasattr(expected, "__text_signature__"))
        self.assertEqual(set(actual.__dict__), set(expected.__dict__))
        self.assertEqual(actual.__code__.co_freevars, expected.__code__.co_freevars)
        self.assertEqual(actual.__code__.co_cellvars, expected.__code__.co_cellvars)

        wildcard_supported = {
            "Attribute",
            "annotate",
            "export",
            "ignore",
            "isinstance",
            "onednn_fusion_enabled",
            "script_if_tracing",
            "strict_fusion",
            "unused",
        }
        public_supported = {
            *wildcard_supported,
            "is_scripting",
            "is_tracing",
            "optimized_execution",
        }
        self.assertNotIn("optimized_execution", actual_jit.__all__)
        self.assertNotIn("optimized_execution", expected_jit.__all__)
        self.assertEqual(
            actual_jit.__all__,
            [
                name
                for name in expected_jit.__all__
                if name in wildcard_supported
            ],
        )
        self.assertEqual(
            {name for name in vars(actual_jit) if not name.startswith("_")},
            public_supported,
        )

        actual_namespace = {}
        expected_namespace = {}
        actual_wildcard = {}
        expected_wildcard = {}
        exec("from torch_rs.jit import optimized_execution", actual_namespace)
        exec("from torch.jit import optimized_execution", expected_namespace)
        exec("from torch_rs.jit import *", actual_wildcard)
        exec("from torch.jit import *", expected_wildcard)
        self.assertIs(actual_namespace["optimized_execution"], actual)
        self.assertIs(expected_namespace["optimized_execution"], expected)
        self.assertNotIn("optimized_execution", actual_wildcard)
        self.assertNotIn("optimized_execution", expected_wildcard)

        self.assertEqual(
            {name for name in vars(actual_fuser) if not name.startswith("_")},
            {"optimized_execution"},
        )
        self.assertIn("optimized_execution", vars(expected_fuser))
        self.assertFalse(hasattr(actual_fuser, "fuser"))
        self.assertFalse(hasattr(actual_fuser, "set_fusion_strategy"))
        self.assertFalse(hasattr(actual_fuser, "last_executed_optimized_graph"))

        self.assertEqual(copy.copy(actual) is actual, copy.copy(expected) is expected)
        self.assertEqual(
            copy.deepcopy(actual) is actual,
            copy.deepcopy(expected) is expected,
        )
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(kind="function", protocol=protocol):
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(pickle.loads(pickle.dumps(expected, protocol)), expected)

        actual_context = actual(False)
        expected_context = expected(False)
        self.assertEqual(type(actual_context), type(expected_context))
        self.assertEqual(actual_context.__doc__, expected_context.__doc__)
        self.assertEqual(
            actual_context.func.__module__.replace("torch_rs", "torch"),
            expected_context.func.__module__,
        )
        self.assertEqual(actual_context.args, expected_context.args)
        self.assertEqual(actual_context.kwds, expected_context.kwds)
        self.assertEqual(
            copy.copy(actual_context).gen is actual_context.gen,
            copy.copy(expected_context).gen is expected_context.gen,
        )
        for call in (
            lambda: copy.deepcopy(actual_context),
            lambda: pickle.dumps(actual_context),
        ):
            with self.subTest(call=call):
                with self.assertRaises(TypeError):
                    call()

        old_actual = actual
        active_context = old_actual(True)
        active_context.__enter__()
        importlib.reload(actual_fuser)
        self.assertIs(actual_jit.optimized_execution, old_actual)
        self.assertIs(active_context.__exit__(None, None, None), False)
        importlib.reload(actual_jit)
        self.assertIs(actual_jit.optimized_execution, actual_fuser.optimized_execution)

        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(old_actual)
        self.assertIs(
            pickle.loads(pickle.dumps(actual_jit.optimized_execution)),
            actual_jit.optimized_execution,
        )

    def test_scripting_tracing_graph_executor_and_parallelism_stay_outside_scope(self):
        expected_fuser = importlib.import_module("torch.jit._fuser")

        self.assertTrue(callable(torch.jit.optimized_execution))
        self.assertTrue(callable(reference_torch.jit.optimized_execution))
        self.assertIs(torch.jit.is_scripting(), False)
        self.assertIs(torch.jit.is_tracing(), False)
        expected_public = {
            name for name in vars(reference_torch.jit) if not name.startswith("_")
        }
        for name in (
            "CompilationUnit",
            "ScriptFunction",
            "ScriptModule",
            "fork",
            "script",
            "script_method",
            "trace",
            "trace_module",
            "wait",
        ):
            with self.subTest(name=name):
                self.assertIn(name, expected_public)
                self.assertFalse(hasattr(torch.jit, name))
        for name in ("fuser", "last_executed_optimized_graph", "set_fusion_strategy"):
            with self.subTest(name=name):
                self.assertIn(name, vars(expected_fuser))
                self.assertFalse(hasattr(torch.jit._fuser, name))
        self.assertFalse(hasattr(torch._C, "_get_graph_executor_optimize"))
        self.assertFalse(hasattr(torch._C, "_set_graph_executor_optimize"))
        self.assertFalse(hasattr(torch, "compile"))


if __name__ == "__main__":
    unittest.main()
