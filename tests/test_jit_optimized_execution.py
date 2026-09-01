import contextlib
import copy
import decimal
import fractions
import importlib
import inspect
import numbers
import pickle
import re
import subprocess
import sys
import threading
import types
import typing
import unittest

import numpy as np
import torch_rs as torch


FUNCTION_DOC = "Context manager that controls whether the JIT's executor will run optimizations before executing a function."


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


class JitOptimizedExecutionTests(unittest.TestCase):
    def test_boolean_like_arguments_are_validated_on_context_entry(self):
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
            torch.tensor(0.0),
            torch.tensor([1.0]),
        )
        for value in accepted:
            with self.subTest(value=value):
                context = torch.jit.optimized_execution(value)
                self.assertEqual(type(context).__module__, "contextlib")
                self.assertIsNone(context.__enter__())
                self.assertIs(context.__exit__(None, None, None), False)

        for result in (False, True):
            with self.subTest(custom_bool=result):
                log = []
                context = torch.jit.optimized_execution(
                    _BoolProbe(str(result), log, result=result)
                )
                self.assertEqual(log, [])
                with context as entered:
                    self.assertIsNone(entered)
                self.assertEqual(log, [str(result)])

    def test_invalid_arguments_raise_on_entry_without_booling_containers(self):
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
            torch.tensor([]),
            torch.tensor([1.0, 2.0]),
        )
        for value in invalid:
            with self.subTest(value_type=type(value).__name__):
                context = torch.jit.optimized_execution(value)
                with self.assertRaises(TypeError) as raised:
                    context.__enter__()
                message = str(raised.exception)
                self.assertIn(
                    "_set_graph_executor_optimize(): incompatible function arguments",
                    message,
                )
                self.assertIn("1. (arg0: bool) -> None", message)
                self.assertIn(f"Invoked with: {value!r}", message)

        error = _ContextBodyError("truthiness failed")
        probe = _BoolProbe("raising", [], error=error)
        context = torch.jit.optimized_execution(probe)
        with self.assertRaises(TypeError) as raised:
            context.__enter__()
        self.assertIn(
            "_set_graph_executor_optimize(): incompatible function arguments",
            str(raised.exception),
        )
        self.assertEqual(probe.log, ["raising"])

    def test_binding_errors_match_pytorch_2_13(self):
        function = torch.jit.optimized_execution
        cases = (
            (
                lambda: function(),
                "optimized_execution() missing 1 required positional argument: 'should_optimize'",
            ),
            (
                lambda: function(True, False),
                "optimized_execution() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: function(enabled=True),
                "optimized_execution() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: function(True, should_optimize=False),
                "optimized_execution() got multiple values for argument 'should_optimize'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

        self.assertIsNone(function(should_optimize=True).__enter__())

    def test_nested_exceptional_and_grad_state_preservation(self):
        states = [torch.is_grad_enabled()]
        with torch.jit.optimized_execution(True) as first_entered:
            states.append(torch.is_grad_enabled())
            with torch.jit.optimized_execution(False) as nested_entered:
                states.append(torch.is_grad_enabled())
            states.append(torch.is_grad_enabled())
        states.append(torch.is_grad_enabled())

        with torch.no_grad():
            states.append(torch.is_grad_enabled())
            with torch.jit.optimized_execution(False) as disabled_entered:
                states.append(torch.is_grad_enabled())
                with torch.jit.optimized_execution(True):
                    states.append(torch.is_grad_enabled())
            states.append(torch.is_grad_enabled())
        states.append(torch.is_grad_enabled())

        marker = _ContextBodyError("body failed")
        with self.assertRaises(_ContextBodyError) as raised:
            with torch.jit.optimized_execution(True) as exceptional_entered:
                self.assertIsNone(exceptional_entered)
                raise marker
        self.assertIs(raised.exception, marker)
        self.assertEqual(
            states,
            [True, True, True, True, True, False, False, False, False, True],
        )
        self.assertIsNone(first_entered)
        self.assertIsNone(nested_entered)
        self.assertIsNone(disabled_entered)

    def test_thread_local_grad_state_is_not_modified(self):
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        outcomes = [None] * worker_count
        errors = []

        def worker(index):
            try:
                grad_context = (
                    torch.no_grad() if index % 2 else contextlib.nullcontext()
                )
                with grad_context:
                    barrier.wait(timeout=10)
                    before = torch.is_grad_enabled()
                    with torch.jit.optimized_execution(index % 3 == 0) as entered:
                        inside = torch.is_grad_enabled()
                    outcomes[index] = (
                        before,
                        inside,
                        torch.is_grad_enabled(),
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

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(
            outcomes,
            [
                (True, True, True, None),
                (False, False, False, None),
                (True, True, True, None),
                (False, False, False, None),
                (True, True, True, None),
                (False, False, False, None),
                (True, True, True, None),
                (False, False, False, None),
            ],
        )
        self.assertIs(torch.is_grad_enabled(), True)

    def test_context_is_a_reusable_decorator_factory(self):
        observations = []

        decorator = torch.jit.optimized_execution(False)

        @decorator
        def decorated(value: int, scale: int = 2, *, label: str = "x") -> tuple:
            """decorated doc"""
            observations.append((torch.is_grad_enabled(), value, scale, label))
            if label == "raise":
                raise _ContextBodyError("decorated body failed")
            return value * scale, label

        self.assertEqual(decorated(3, label="first"), (6, "first"))
        with torch.no_grad():
            self.assertEqual(decorated(4, 5), (20, "x"))
        with self.assertRaisesRegex(_ContextBodyError, "decorated body failed"):
            decorated(1, label="raise")
        self.assertEqual(
            observations,
            [
                (True, 3, 2, "first"),
                (False, 4, 5, "x"),
                (True, 1, 2, "raise"),
            ],
        )
        self.assertEqual(decorated.__name__, "decorated")
        self.assertEqual(decorated.__qualname__.split(".")[-1], "decorated")
        self.assertEqual(decorated.__doc__, "decorated doc")
        self.assertEqual(typing.get_type_hints(decorated)["return"], tuple)
        self.assertEqual(
            str(inspect.signature(decorated)),
            str(inspect.signature(decorated.__wrapped__)),
        )

    def test_metadata_imports_copy_pickle_and_reload(self):
        jit = importlib.import_module("torch_rs.jit")
        fuser = importlib.import_module("torch_rs.jit._fuser")
        function = jit.optimized_execution
        wrapped = function.__wrapped__

        self.assertIs(torch.jit, jit)
        self.assertIs(jit._fuser, fuser)
        self.assertIs(sys.modules["torch_rs.jit._fuser"], fuser)
        self.assertIs(function, fuser.optimized_execution)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "(should_optimize)")
        self.assertEqual(function.__annotations__, {})
        self.assertEqual(typing.get_type_hints(function), {})
        self.assertEqual(function.__name__, "optimized_execution")
        self.assertEqual(function.__qualname__, "optimized_execution")
        self.assertEqual(function.__module__, "torch_rs.jit._fuser")
        self.assertIs(inspect.getmodule(function), fuser)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {"__wrapped__": wrapped})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(function.__code__.co_freevars, ("func",))
        self.assertEqual(function.__code__.co_cellvars, ())

        self.assertIs(type(wrapped), types.FunctionType)
        self.assertEqual(str(inspect.signature(wrapped)), "(should_optimize)")
        self.assertEqual(wrapped.__annotations__, {})
        self.assertEqual(wrapped.__name__, "optimized_execution")
        self.assertEqual(wrapped.__qualname__, "optimized_execution")
        self.assertEqual(wrapped.__module__, "torch_rs.jit._fuser")
        self.assertIs(inspect.getmodule(wrapped), fuser)
        self.assertEqual(wrapped.__doc__, FUNCTION_DOC)
        self.assertIsNone(wrapped.__defaults__)
        self.assertIsNone(wrapped.__kwdefaults__)
        self.assertEqual(wrapped.__dict__, {})

        self.assertNotIn("optimized_execution", jit.__all__)
        self.assertEqual(
            {name for name in vars(jit) if not name.startswith("_")},
            {
                "Attribute",
                "annotate",
                "export",
                "ignore",
                "isinstance",
                "is_scripting",
                "is_tracing",
                "onednn_fusion_enabled",
                "optimized_execution",
                "script_if_tracing",
                "strict_fusion",
                "unused",
            },
        )
        self.assertFalse(hasattr(fuser, "__all__"))
        self.assertEqual(
            {name for name in vars(fuser) if not name.startswith("_")},
            {"optimized_execution"},
        )

        explicit_namespace = {}
        fuser_namespace = {}
        jit_wildcard_namespace = {}
        fuser_wildcard_namespace = {}
        top_level_namespace = {}
        exec("from torch_rs.jit import optimized_execution", explicit_namespace)
        exec("from torch_rs.jit._fuser import optimized_execution", fuser_namespace)
        exec("from torch_rs.jit import *", jit_wildcard_namespace)
        exec("from torch_rs.jit._fuser import *", fuser_wildcard_namespace)
        exec("from torch_rs import *", top_level_namespace)
        self.assertIs(explicit_namespace["optimized_execution"], function)
        self.assertIs(fuser_namespace["optimized_execution"], function)
        self.assertNotIn("optimized_execution", jit_wildcard_namespace)
        self.assertIs(fuser_wildcard_namespace["optimized_execution"], function)
        self.assertNotIn("optimized_execution", top_level_namespace)
        self.assertFalse(hasattr(torch, "optimized_execution"))

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(kind="function", protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.jit._fuser", payload)
                self.assertIs(pickle.loads(payload), function)

        context = function(False)
        self.assertEqual(type(context).__module__, "contextlib")
        self.assertEqual(type(context).__qualname__, "_GeneratorContextManager")
        self.assertEqual(context.__doc__, FUNCTION_DOC)
        self.assertIs(context.func, wrapped)
        self.assertEqual(context.args, (False,))
        self.assertEqual(context.kwds, {})
        copied = copy.copy(context)
        self.assertIsNot(copied, context)
        self.assertIs(copied.gen, context.gen)
        with self.assertRaisesRegex(TypeError, "cannot pickle 'generator' object"):
            copy.deepcopy(context)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(kind="context", protocol=protocol):
                with self.assertRaisesRegex(
                    TypeError,
                    "cannot pickle 'generator' object",
                ):
                    pickle.dumps(context, protocol=protocol)

        old_function = function
        old_wrapped = wrapped
        namespace = fuser.__dict__
        active_context = old_function(True)
        self.assertIsNone(active_context.__enter__())
        reloaded_fuser = importlib.reload(fuser)

        self.assertIs(reloaded_fuser, fuser)
        self.assertIs(fuser.__dict__, namespace)
        self.assertIsNot(fuser.optimized_execution, old_function)
        self.assertIsNot(fuser.optimized_execution.__wrapped__, old_wrapped)
        self.assertIs(jit.optimized_execution, old_function)
        self.assertIs(active_context.__exit__(None, None, None), False)

        reloaded_jit = importlib.reload(jit)
        self.assertIs(reloaded_jit, jit)
        self.assertIs(jit.optimized_execution, fuser.optimized_execution)
        self.assertIs(torch.jit, jit)

        with self.assertRaises(pickle.PicklingError) as raised:
            pickle.dumps(old_function)
        message = re.sub(r"0x[0-9a-fA-F]+", "0x...", str(raised.exception))
        self.assertEqual(
            message,
            "Can't pickle <function optimized_execution at 0x...>: it's not "
            "the same object as torch_rs.jit._fuser.optimized_execution",
        )
        self.assertIs(
            pickle.loads(pickle.dumps(jit.optimized_execution)),
            jit.optimized_execution,
        )

    def test_scripting_tracing_graph_executor_and_parallelism_remain_unsupported(self):
        self.assertTrue(callable(torch.jit.optimized_execution))
        self.assertIs(torch.jit.is_scripting(), False)
        self.assertIs(torch.jit.is_tracing(), False)
        for name in (
            "CompilationUnit",
            "ScriptFunction",
            "ScriptModule",
            "enable_onednn_fusion",
            "fork",
            "fuser",
            "last_executed_optimized_graph",
            "script",
            "script_method",
            "set_fusion_strategy",
            "trace",
            "trace_module",
            "wait",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.jit, name))
        for name in (
            "fuser",
            "last_executed_optimized_graph",
            "set_fusion_strategy",
        ):
            with self.subTest(fuser_name=name):
                self.assertFalse(hasattr(torch.jit._fuser, name))
        self.assertFalse(hasattr(torch._C, "_get_graph_executor_optimize"))
        self.assertFalse(hasattr(torch._C, "_set_graph_executor_optimize"))
        self.assertFalse(hasattr(torch, "compile"))

    def test_importing_the_package_does_not_import_pytorch(self):
        script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

with torch.jit.optimized_execution(True) as entered:
    assert entered is None

@torch.jit.optimized_execution(False)
def function(value):
    return value + 1

assert function(3) == 4
assert torch.jit.optimized_execution.__module__ == "torch_rs.jit._fuser"
assert not hasattr(torch.jit, "script")
assert not hasattr(torch.jit, "trace")
assert not hasattr(torch.jit, "fork")
assert not hasattr(torch.jit, "wait")
assert not hasattr(torch._C, "_set_graph_executor_optimize")
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
