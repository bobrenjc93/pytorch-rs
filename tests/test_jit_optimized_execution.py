import copy
import enum
import importlib
import inspect
import pickle
import re
import subprocess
import sys
import threading
import types
import unittest

import torch_rs as torch


OPTIMIZED_EXECUTION_DOC = (
    "Context manager that controls whether the JIT's executor will run "
    "optimizations before executing a function."
)


class _ContextBodyError(Exception):
    pass


class _RejectTruthiness:
    def __init__(self):
        self.calls = 0

    def __bool__(self):
        self.calls += 1
        raise AssertionError("truth-value error should be replaced")


class _PlainEnum(enum.Enum):
    ITEM = 1


class _FailingRepr:
    def __repr__(self):
        raise RuntimeError("repr exploded")


class _FailingBoolAndRepr:
    def __bool__(self):
        raise RuntimeError("bool exploded")

    def __repr__(self):
        raise LookupError("repr exploded")


class _HideBoolMeta(type):
    def __getattribute__(cls, name):
        if name == "__bool__":
            raise AttributeError(name)
        return super().__getattribute__(name)


class _TruthValue:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __bool__(self):
        self.calls += 1
        return self.value


class _MetaclassHiddenTruthValue(_TruthValue, metaclass=_HideBoolMeta):
    pass


class _InstanceHiddenTruthValue(_TruthValue):
    def __getattribute__(self, name):
        if name == "__bool__":
            raise AttributeError(name)
        return super().__getattribute__(name)


class _StateChangingTruthValue:
    def __init__(self, root, intermediate, value):
        self.root = root
        self.intermediate = intermediate
        self.value = value

    def __bool__(self):
        self.root._C._set_graph_executor_optimize(self.intermediate)
        return self.value


class OptimizedExecutionTests(unittest.TestCase):
    def setUp(self):
        self.fuser = importlib.import_module("torch_rs.jit._fuser")
        self.original = torch._C._get_graph_executor_optimize()
        torch._C._set_graph_executor_optimize(True)

    def tearDown(self):
        torch._C._set_graph_executor_optimize(self.original)

    def test_entry_nesting_and_exception_safe_restoration(self):
        optimized_execution = self.fuser.optimized_execution

        for initial in (False, True):
            for requested in (False, True, None, 0, 1):
                with self.subTest(initial=initial, requested=requested):
                    torch._C._set_graph_executor_optimize(initial)
                    context = optimized_execution(requested)
                    self.assertIs(
                        torch._C._get_graph_executor_optimize(), initial
                    )
                    self.assertIsNone(context.__enter__())
                    self.assertIs(
                        torch._C._get_graph_executor_optimize(),
                        bool(requested),
                    )
                    self.assertIs(context.__exit__(None, None, None), False)
                    self.assertIs(
                        torch._C._get_graph_executor_optimize(), initial
                    )

        torch._C._set_graph_executor_optimize(True)
        with optimized_execution(False) as outer:
            self.assertIsNone(outer)
            self.assertIs(torch._C._get_graph_executor_optimize(), False)
            with optimized_execution(True) as inner:
                self.assertIsNone(inner)
                self.assertIs(torch._C._get_graph_executor_optimize(), True)
            self.assertIs(torch._C._get_graph_executor_optimize(), False)
        self.assertIs(torch._C._get_graph_executor_optimize(), True)

        marker = _ContextBodyError("body failed")
        with self.assertRaises(_ContextBodyError) as raised:
            with optimized_execution(False) as entered:
                self.assertIsNone(entered)
                self.assertIs(torch._C._get_graph_executor_optimize(), False)
                raise marker
        self.assertIs(raised.exception, marker)
        self.assertIs(torch._C._get_graph_executor_optimize(), True)

    def test_argument_validation_is_deferred_until_context_entry(self):
        optimized_execution = self.fuser.optimized_execution
        rejected = _RejectTruthiness()
        invalid_values = (
            "",
            "enabled",
            [],
            object(),
            _PlainEnum.ITEM,
            _FailingRepr(),
            _FailingBoolAndRepr(),
            rejected,
        )

        for state in (False, True):
            for value in invalid_values:
                with self.subTest(state=state, value_type=type(value).__name__):
                    torch._C._set_graph_executor_optimize(state)
                    context = optimized_execution(value)
                    self.assertIs(
                        torch._C._get_graph_executor_optimize(), state
                    )
                    with self.assertRaises(TypeError) as raised:
                        context.__enter__()
                    message = re.sub(
                        r"0x[0-9a-fA-F]+", "0x...", str(raised.exception)
                    )
                    try:
                        rendered = repr(value)
                    except Exception:
                        rendered = "<repr raised Error>"
                    rendered = re.sub(r"0x[0-9a-fA-F]+", "0x...", rendered)
                    self.assertEqual(
                        message,
                        "_set_graph_executor_optimize(): incompatible function "
                        "arguments. The following argument types are supported:\n"
                        "    1. (arg0: bool) -> None\n\n"
                        f"Invoked with: {rendered}",
                    )
                    self.assertEqual(raised.exception.args, (str(raised.exception),))
                    self.assertIs(
                        torch._C._get_graph_executor_optimize(), state
                    )

        self.assertEqual(rejected.calls, 2)

        for value, expected in (
            (0.0, False),
            (1.5, True),
            (range(0), False),
            (range(1), True),
            (_TruthValue(False), False),
            (_TruthValue(True), True),
            (_MetaclassHiddenTruthValue(False), False),
            (_MetaclassHiddenTruthValue(True), True),
            (_InstanceHiddenTruthValue(False), False),
            (_InstanceHiddenTruthValue(True), True),
        ):
            with self.subTest(coerced_type=type(value).__name__, expected=expected):
                torch._C._set_graph_executor_optimize(not expected)
                with optimized_execution(value) as entered:
                    self.assertIsNone(entered)
                    self.assertIs(
                        torch._C._get_graph_executor_optimize(), expected
                    )
                self.assertIs(
                    torch._C._get_graph_executor_optimize(), not expected
                )
                if isinstance(value, _TruthValue):
                    self.assertEqual(value.calls, 1)

        torch._C._set_graph_executor_optimize(True)
        cases = (
            (
                lambda: optimized_execution(),
                "optimized_execution() missing 1 required positional argument: "
                "'should_optimize'",
            ),
            (
                lambda: optimized_execution(True, False),
                "optimized_execution() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: optimized_execution(enabled=True),
                "optimized_execution() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: optimized_execution(False, should_optimize=True),
                "optimized_execution() got multiple values for argument "
                "'should_optimize'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertIs(torch._C._get_graph_executor_optimize(), True)

    def test_context_is_a_reusable_decorator_factory(self):
        observations = []

        @self.fuser.optimized_execution(False)
        def decorated(value):
            observations.append(torch._C._get_graph_executor_optimize())
            if value == "raise":
                raise _ContextBodyError("decorated body failed")
            return value

        self.assertEqual(decorated("first"), "first")
        self.assertEqual(decorated("second"), "second")
        with self.assertRaisesRegex(_ContextBodyError, "decorated body failed"):
            decorated("raise")
        self.assertEqual(observations, [False, False, False])
        self.assertIs(torch._C._get_graph_executor_optimize(), True)

    def test_optimization_preference_is_thread_local(self):
        torch._C._set_graph_executor_optimize(False)
        worker_entered = threading.Event()
        main_changed = threading.Event()
        observations = []
        errors = []

        def worker():
            try:
                observations.append(("worker-default", torch._C._get_graph_executor_optimize()))
                with self.fuser.optimized_execution(False) as entered:
                    observations.append(
                        ("worker-enter", entered, torch._C._get_graph_executor_optimize())
                    )
                    worker_entered.set()
                    if not main_changed.wait(timeout=10):
                        raise RuntimeError("timed out waiting for main thread")
                    observations.append(
                        ("worker-resume", torch._C._get_graph_executor_optimize())
                    )
                observations.append(("worker-exit", torch._C._get_graph_executor_optimize()))
            except BaseException as error:
                errors.append(error)
                worker_entered.set()

        thread = threading.Thread(target=worker)
        thread.start()
        try:
            self.assertTrue(worker_entered.wait(timeout=10))
            self.assertEqual(errors, [])
            self.assertIs(torch._C._get_graph_executor_optimize(), False)
            torch._C._set_graph_executor_optimize(True)
        finally:
            main_changed.set()
            thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(
            observations,
            [
                ("worker-default", True),
                ("worker-enter", None, False),
                ("worker-resume", False),
                ("worker-exit", True),
            ],
        )
        self.assertIs(torch._C._get_graph_executor_optimize(), True)

    def test_metadata_imports_copying_and_pickling(self):
        fuser = self.fuser
        jit = torch.jit
        function = fuser.optimized_execution
        wrapped = function.__wrapped__

        self.assertIs(jit.optimized_execution, function)
        self.assertIs(sys.modules["torch_rs.jit._fuser"], fuser)
        self.assertIs(type(fuser), types.ModuleType)
        self.assertIsNone(fuser.__doc__)
        self.assertFalse(hasattr(fuser, "__all__"))
        self.assertEqual(
            {name for name in vars(fuser) if not name.startswith("_")},
            {"contextlib", "optimized_execution", "torch", "warnings"},
        )

        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "(should_optimize)")
        self.assertEqual(inspect.get_annotations(function), {})
        self.assertEqual(function.__name__, "optimized_execution")
        self.assertEqual(function.__qualname__, "optimized_execution")
        self.assertEqual(function.__module__, "torch_rs.jit._fuser")
        self.assertIs(inspect.getmodule(function), fuser)
        self.assertEqual(function.__doc__, OPTIMIZED_EXECUTION_DOC)
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {"__wrapped__": wrapped})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(function.__code__.co_names, ("_GeneratorContextManager",))
        self.assertEqual(function.__code__.co_freevars, ("func",))
        self.assertEqual(function.__code__.co_cellvars, ())

        self.assertIs(type(wrapped), types.FunctionType)
        self.assertEqual(str(inspect.signature(wrapped)), "(should_optimize)")
        self.assertEqual(inspect.get_annotations(wrapped), {})
        self.assertEqual(wrapped.__name__, "optimized_execution")
        self.assertEqual(wrapped.__qualname__, "optimized_execution")
        self.assertEqual(wrapped.__module__, "torch_rs.jit._fuser")
        self.assertIs(inspect.getmodule(wrapped), fuser)
        self.assertEqual(wrapped.__doc__, OPTIMIZED_EXECUTION_DOC)
        self.assertIsNone(wrapped.__defaults__)
        self.assertIsNone(wrapped.__kwdefaults__)
        self.assertEqual(wrapped.__dict__, {})
        self.assertEqual(
            wrapped.__code__.co_names,
            (
                "torch",
                "_C",
                "_get_graph_executor_optimize",
                "_set_graph_executor_optimize",
            ),
        )
        self.assertEqual(wrapped.__code__.co_freevars, ())
        self.assertEqual(wrapped.__code__.co_cellvars, ())

        package_import = {}
        module_import = {}
        package_wildcard = {}
        module_wildcard = {}
        exec("from torch_rs.jit import optimized_execution", package_import)
        exec("from torch_rs.jit._fuser import optimized_execution", module_import)
        exec("from torch_rs.jit import *", package_wildcard)
        exec("from torch_rs.jit._fuser import *", module_wildcard)
        self.assertIs(package_import["optimized_execution"], function)
        self.assertIs(module_import["optimized_execution"], function)
        self.assertNotIn("optimized_execution", package_wildcard)
        self.assertIs(module_wildcard["optimized_execution"], function)
        self.assertEqual(
            {name for name in module_wildcard if not name.startswith("__")},
            {"contextlib", "optimized_execution", "torch", "warnings"},
        )
        self.assertNotIn("optimized_execution", jit.__all__)
        self.assertNotIn("optimized_execution", torch.__all__)
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
        self.assertEqual(context.__doc__, OPTIMIZED_EXECUTION_DOC)
        self.assertIs(context.func, wrapped)
        self.assertEqual(context.args, (False,))
        self.assertEqual(context.kwds, {})
        shallow = copy.copy(context)
        self.assertIsNot(shallow, context)
        self.assertIs(shallow.gen, context.gen)
        with self.assertRaisesRegex(TypeError, "cannot pickle 'generator' object"):
            copy.deepcopy(context)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(kind="context", protocol=protocol):
                with self.assertRaisesRegex(
                    TypeError, "cannot pickle 'generator' object"
                ):
                    pickle.dumps(context, protocol=protocol)

    def test_reload_preserves_state_and_existing_contexts(self):
        fuser = self.fuser
        jit = torch.jit
        old_function = fuser.optimized_execution
        old_wrapped = old_function.__wrapped__
        namespace = fuser.__dict__
        active_context = old_function(False)

        self.assertIsNone(active_context.__enter__())
        self.assertIs(torch._C._get_graph_executor_optimize(), False)
        reloaded = importlib.reload(fuser)

        self.assertIs(reloaded, fuser)
        self.assertIs(fuser.__dict__, namespace)
        self.assertIs(sys.modules[fuser.__name__], fuser)
        self.assertIsNot(fuser.optimized_execution, old_function)
        self.assertIsNot(fuser.optimized_execution.__wrapped__, old_wrapped)
        self.assertIs(jit.optimized_execution, old_function)
        self.assertIs(torch._C._get_graph_executor_optimize(), False)
        self.assertIs(active_context.__exit__(None, None, None), False)
        self.assertIs(torch._C._get_graph_executor_optimize(), True)

        for function in (old_function, fuser.optimized_execution):
            with self.subTest(function=function):
                with function(False) as entered:
                    self.assertIsNone(entered)
                    self.assertIs(torch._C._get_graph_executor_optimize(), False)
                self.assertIs(torch._C._get_graph_executor_optimize(), True)

        with self.assertRaises(pickle.PicklingError) as raised:
            pickle.dumps(old_function)
        message = re.sub(r"0x[0-9a-fA-F]+", "0x...", str(raised.exception))
        self.assertEqual(
            message,
            "Can't pickle <function optimized_execution at 0x...>: it's not "
            "the same object as torch_rs.jit._fuser.optimized_execution",
        )
        self.assertIs(
            pickle.loads(pickle.dumps(fuser.optimized_execution)),
            fuser.optimized_execution,
        )

        self.assertIs(importlib.reload(jit), jit)
        self.assertIs(jit.optimized_execution, fuser.optimized_execution)

    def test_native_preference_accessors_and_eager_execution_are_independent(self):
        self.assertTrue(hasattr(torch._C, "_get_graph_executor_optimize"))
        self.assertTrue(hasattr(torch._C, "_set_graph_executor_optimize"))
        self.assertNotIn("_get_graph_executor_optimize", torch._C.__all__)
        self.assertNotIn("_set_graph_executor_optimize", torch._C.__all__)
        self.assertFalse(hasattr(torch, "_get_graph_executor_optimize"))
        self.assertFalse(hasattr(torch, "_set_graph_executor_optimize"))

        self.assertIs(torch._C._get_graph_executor_optimize(False), True)
        self.assertIs(torch._C._get_graph_executor_optimize(), False)
        self.assertIs(torch._C._get_graph_executor_optimize(True), False)
        self.assertIs(torch._C._get_graph_executor_optimize(), True)

        changing_value = _StateChangingTruthValue(torch, False, True)
        self.assertIs(
            torch._C._get_graph_executor_optimize(changing_value), False
        )
        self.assertIs(torch._C._get_graph_executor_optimize(), True)

        for should_optimize in (False, True):
            with self.subTest(should_optimize=should_optimize):
                leaf = torch.tensor([2.0, 3.0], requires_grad=True)
                with self.fuser.optimized_execution(should_optimize):
                    output = leaf * 4.0
                    self.assertEqual(output.tolist(), [8.0, 12.0])
                    output.sum().backward()
                self.assertEqual(leaf.grad.tolist(), [4.0, 4.0])
                self.assertIs(torch.is_grad_enabled(), True)

        with torch.no_grad():
            self.assertIs(torch.is_grad_enabled(), False)
            with self.fuser.optimized_execution(False) as entered:
                self.assertIsNone(entered)
                self.assertIs(torch.is_grad_enabled(), False)
        self.assertIs(torch.is_grad_enabled(), True)

    def test_fresh_process_does_not_import_pytorch(self):
        script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch
from torch_rs.jit._fuser import optimized_execution

assert torch._C._get_graph_executor_optimize() is True
with optimized_execution(False) as entered:
    assert entered is None
    assert torch._C._get_graph_executor_optimize() is False
    leaf = torch.tensor([2.0], requires_grad=True)
    (leaf * 3.0).sum().backward()
    assert leaf.grad.tolist() == [3.0]
assert torch._C._get_graph_executor_optimize() is True
assert "torch" not in sys.modules
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
