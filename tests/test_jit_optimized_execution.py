import contextlib
import copy
import importlib
import inspect
import pathlib
import pickle
import subprocess
import sys
import threading
import types
import typing
import unittest

import torch_rs as torch


FUNCTION_DOC = (
    "Context manager that controls whether the JIT's executor will run "
    "optimizations before executing a function."
)
OPTIMIZE_TYPE_ERROR_PREFIX = (
    "_set_graph_executor_optimize(): incompatible function arguments. "
    "The following argument types are supported:\n"
    "    1. (arg0: bool) -> None\n\n"
    "Invoked with: "
)
ROOT = pathlib.Path(__file__).resolve().parents[1]


@torch.jit.optimized_execution(True)
def _picklable_optimized_execution_function(value):
    return value


class BoolLike:
    def __init__(self, value, events=None, label=None):
        self.value = value
        self.events = events
        self.label = label

    def __bool__(self):
        if self.events is not None:
            self.events.append(self.label)
        return self.value

    def __repr__(self):
        return f"BoolLike({self.value!r})"


class BadBool:
    def __bool__(self):
        raise RuntimeError("bool failed")

    def __repr__(self):
        return "BadBool()"


class BadRepr:
    def __repr__(self):
        raise RuntimeError("repr failed")


class BadBoolBadRepr:
    def __bool__(self):
        raise RuntimeError("bool failed")

    def __repr__(self):
        raise RuntimeError("repr failed")


class KeyboardInterruptBool:
    def __bool__(self):
        raise KeyboardInterrupt("bool interrupted")

    def __repr__(self):
        return "KeyboardInterruptBool()"


class SystemExitBool:
    def __bool__(self):
        raise SystemExit("bool exited")

    def __repr__(self):
        return "SystemExitBool()"


class LenOnly:
    def __len__(self):
        return 1

    def __repr__(self):
        return "LenOnly()"


class JitOptimizedExecutionTests(unittest.TestCase):
    def assert_invalid_entry(self, value, expected_suffix):
        context = torch.jit.optimized_execution(value)
        with self.assertRaises(TypeError) as raised:
            context.__enter__()
        message = OPTIMIZE_TYPE_ERROR_PREFIX + expected_suffix
        self.assertEqual(str(raised.exception), message)
        self.assertEqual(raised.exception.args, (message,))
        self.assertIs(context.__exit__(None, None, None), False)

    def test_truthy_falsey_contexts_are_eager_no_ops(self):
        events = []
        cases = (
            (True, ()),
            (False, ()),
            (1, ()),
            (0, ()),
            (None, ()),
            (BoolLike(True, events, "truthy object"), ("truthy object",)),
            (BoolLike(False, events, "falsey object"), ("falsey object",)),
        )
        for value, expected_new_events in cases:
            with self.subTest(value=value):
                before_events = tuple(events)
                context = torch.jit.optimized_execution(value)
                self.assertIs(type(context), contextlib._GeneratorContextManager)
                self.assertEqual(
                    {
                        name
                        for name in context.__dict__
                        if name != "gen"
                    },
                    {"func", "args", "kwds", "__doc__"},
                )
                self.assertEqual(context.__dict__["args"], (value,))
                self.assertEqual(context.__dict__["kwds"], {})
                self.assertIs(
                    context.__dict__["func"],
                    torch.jit.optimized_execution.__wrapped__,
                )
                self.assertEqual(context.__dict__["__doc__"], FUNCTION_DOC)
                self.assertIs(context.__enter__(), None)
                self.assertEqual(
                    tuple(events),
                    before_events + expected_new_events,
                )
                self.assertIs(torch.is_grad_enabled(), True)
                self.assertIs(context.__exit__(None, None, None), False)
                self.assertIs(torch.is_grad_enabled(), True)

    def test_invalid_values_are_rejected_on_context_entry(self):
        invalid_cases = (
            ([], "[]"),
            ([1], "[1]"),
            ("", "''"),
            ("yes", "'yes'"),
            (BadBool(), "BadBool()"),
            (BadRepr(), "<repr raised Error>"),
            (BadBoolBadRepr(), "<repr raised Error>"),
            (KeyboardInterruptBool(), "KeyboardInterruptBool()"),
            (SystemExitBool(), "SystemExitBool()"),
            (LenOnly(), "LenOnly()"),
        )
        for index, (value, expected_suffix) in enumerate(invalid_cases):
            with self.subTest(index=index):
                self.assert_invalid_entry(value, expected_suffix)

    def test_nested_contexts_exceptional_exits_and_grad_state(self):
        events = []
        states = [torch.is_grad_enabled()]
        error = RuntimeError("forwarded failure")

        try:
            with torch.jit.optimized_execution(
                BoolLike(True, events, "outer")
            ) as outer_entered:
                states.append(torch.is_grad_enabled())
                with torch.jit.optimized_execution(
                    BoolLike(False, events, "inner")
                ) as inner_entered:
                    states.append(torch.is_grad_enabled())
                states.append(torch.is_grad_enabled())
                with torch.jit.optimized_execution(
                    BoolLike(True, events, "exception")
                ):
                    raise error
        except RuntimeError as raised:
            self.assertIs(raised, error)
        else:
            self.fail("optimized_execution suppressed the context exception")

        states.append(torch.is_grad_enabled())
        with torch.no_grad():
            states.append(torch.is_grad_enabled())
            with torch.jit.optimized_execution(True) as no_grad_entered:
                states.append(torch.is_grad_enabled())
            states.append(torch.is_grad_enabled())
        states.append(torch.is_grad_enabled())

        self.assertIsNone(outer_entered)
        self.assertIsNone(inner_entered)
        self.assertIsNone(no_grad_entered)
        self.assertEqual(events, ["outer", "inner", "exception"])
        self.assertEqual(
            states,
            [True, True, True, True, True, False, False, False, True],
        )

    def test_context_preserves_thread_local_grad_state(self):
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
                    with torch.jit.optimized_execution(index % 2 == 0) as entered:
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

    def test_decorator_preserves_function_metadata_and_validates_on_call(self):
        events = []
        calls = []

        def function(
            value: int,
            /,
            scale: int = 2,
            *,
            label: str = "value",
        ) -> tuple[int, int, str]:
            """Function documentation."""
            calls.append((value, scale, label))
            return value, scale, label

        sentinel = object()
        function.custom_attribute = sentinel
        wrapped = torch.jit.optimized_execution(
            BoolLike(True, events, "decorator")
        )(function)

        self.assertIs(type(wrapped), types.FunctionType)
        self.assertIsNot(wrapped, function)
        self.assertEqual(wrapped.__name__, function.__name__)
        self.assertEqual(wrapped.__qualname__, function.__qualname__)
        self.assertEqual(wrapped.__module__, function.__module__)
        self.assertEqual(wrapped.__doc__, function.__doc__)
        self.assertEqual(wrapped.__annotations__, function.__annotations__)
        self.assertEqual(typing.get_type_hints(wrapped), typing.get_type_hints(function))
        self.assertIsNone(wrapped.__defaults__)
        self.assertIsNone(wrapped.__kwdefaults__)
        self.assertIs(wrapped.custom_attribute, sentinel)
        self.assertIs(wrapped.__wrapped__, function)
        self.assertIs(inspect.unwrap(wrapped), function)
        self.assertEqual(inspect.signature(wrapped), inspect.signature(function))
        self.assertEqual(set(wrapped.__dict__), {"custom_attribute", "__wrapped__"})
        self.assertEqual(wrapped(3, label="scaled"), (3, 2, "scaled"))
        self.assertEqual(calls, [(3, 2, "scaled")])
        self.assertEqual(events, ["decorator"])

        invalid = torch.jit.optimized_execution("yes")(function)
        with self.assertRaises(TypeError) as raised:
            invalid(4)
        self.assertEqual(str(raised.exception), OPTIMIZE_TYPE_ERROR_PREFIX + "'yes'")
        self.assertEqual(calls, [(3, 2, "scaled")])

    def test_signature_documentation_and_module_ownership(self):
        jit = importlib.import_module("torch_rs.jit")
        fuser = importlib.import_module("torch_rs.jit._fuser")
        function = jit.optimized_execution
        wrapped = function.__wrapped__

        self.assertIs(torch.jit, jit)
        self.assertIs(jit._fuser, fuser)
        self.assertIs(function, fuser.optimized_execution)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "(should_optimize)")
        self.assertEqual(function.__annotations__, {})
        self.assertEqual(typing.get_type_hints(function), {})
        self.assertEqual(function.__name__, "optimized_execution")
        self.assertEqual(function.__qualname__, "optimized_execution")
        self.assertEqual(function.__module__, "torch_rs.jit._fuser")
        self.assertIs(inspect.getmodule(function), fuser)
        self.assertEqual(inspect.cleandoc(function.__doc__), FUNCTION_DOC)
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(set(function.__dict__), {"__wrapped__"})
        self.assertIs(type(wrapped), types.FunctionType)
        self.assertEqual(wrapped.__name__, "optimized_execution")
        self.assertEqual(wrapped.__qualname__, "optimized_execution")
        self.assertEqual(wrapped.__module__, "torch_rs.jit._fuser")
        self.assertIs(inspect.getmodule(wrapped), fuser)
        self.assertEqual(inspect.cleandoc(wrapped.__doc__), FUNCTION_DOC)
        self.assertEqual(str(inspect.signature(wrapped)), "(should_optimize)")
        self.assertEqual(wrapped.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

        keyword_context = function(should_optimize=True)
        self.assertIs(keyword_context.__enter__(), None)
        self.assertIs(keyword_context.__exit__(None, None, None), False)

    def test_imports_wildcards_reload_copy_and_pickle(self):
        jit = torch.jit
        fuser = importlib.import_module("torch_rs.jit._fuser")
        function = jit.optimized_execution
        supported = {
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

        self.assertNotIn("optimized_execution", jit.__all__)
        self.assertEqual(
            {name for name in vars(jit) if not name.startswith("_")},
            {
                *supported,
                "is_scripting",
                "is_tracing",
                "optimized_execution",
            },
        )

        explicit_namespace = {}
        exec("from torch_rs.jit import optimized_execution", explicit_namespace)
        self.assertIs(explicit_namespace["optimized_execution"], function)
        wildcard_namespace = {}
        exec("from torch_rs.jit import *", wildcard_namespace)
        self.assertEqual(
            {name for name in wildcard_namespace if not name.startswith("__")},
            supported,
        )
        self.assertNotIn("optimized_execution", wildcard_namespace)

        fuser_namespace = {}
        exec("from torch_rs.jit._fuser import optimized_execution", fuser_namespace)
        self.assertIs(fuser_namespace["optimized_execution"], function)
        fuser_wildcard_namespace = {}
        exec("from torch_rs.jit._fuser import *", fuser_wildcard_namespace)
        self.assertEqual(
            {name for name in fuser_wildcard_namespace if not name.startswith("__")},
            {"optimized_execution"},
        )

        self.assertNotIn("optimized_execution", torch.__all__)
        self.assertFalse(hasattr(torch, "optimized_execution"))
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("optimized_execution", top_level_namespace)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        self.assertIs(
            copy.copy(_picklable_optimized_execution_function),
            _picklable_optimized_execution_function,
        )
        self.assertIs(
            copy.deepcopy(_picklable_optimized_execution_function),
            _picklable_optimized_execution_function,
        )
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(kind="function", protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.jit._fuser", payload)
                self.assertIs(pickle.loads(payload), function)
            with self.subTest(kind="decorated", protocol=protocol):
                payload = pickle.dumps(
                    _picklable_optimized_execution_function,
                    protocol=protocol,
                )
                self.assertIs(
                    pickle.loads(payload),
                    _picklable_optimized_execution_function,
                )

        context = function(True)
        context.value = {"items": [1, 2]}
        shallow = copy.copy(context)
        self.assertIsNot(shallow, context)
        self.assertIs(type(shallow), type(context))
        self.assertIs(shallow.__dict__["gen"], context.__dict__["gen"])
        self.assertIs(shallow.__dict__["func"], context.__dict__["func"])
        self.assertEqual(shallow.__dict__["args"], (True,))
        self.assertEqual(shallow.__dict__["kwds"], {})
        self.assertIs(shallow.value, context.value)
        with self.assertRaisesRegex(TypeError, "^cannot pickle 'generator' object$"):
            copy.deepcopy(context)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(kind="context", protocol=protocol):
                with self.assertRaisesRegex(
                    TypeError,
                    "^cannot pickle 'generator' object$",
                ):
                    pickle.dumps(context, protocol=protocol)

        original = jit.optimized_execution
        reloaded_fuser = importlib.reload(fuser)
        self.assertIsNot(reloaded_fuser.optimized_execution, original)
        self.assertIs(jit.optimized_execution, original)
        self.assertIs(
            jit._fuser.optimized_execution,
            reloaded_fuser.optimized_execution,
        )
        reloaded_jit = importlib.reload(jit)
        self.assertIs(
            reloaded_jit.optimized_execution,
            reloaded_fuser.optimized_execution,
        )
        self.assertIs(
            torch.jit.optimized_execution,
            reloaded_fuser.optimized_execution,
        )

    def test_call_errors_match_pytorch_2_13_shape(self):
        function = torch.jit.optimized_execution
        cases = (
            (
                lambda: function(),
                "optimized_execution() missing 1 required positional argument: "
                "'should_optimize'",
            ),
            (
                lambda: function(True, False),
                "optimized_execution() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: function(value=True),
                "optimized_execution() got an unexpected keyword argument 'value'",
            ),
            (
                lambda: function(True, should_optimize=False),
                "optimized_execution() got multiple values for argument "
                "'should_optimize'",
            ),
            (
                lambda: function(should_optimize=True, value=False),
                "optimized_execution() got an unexpected keyword argument 'value'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_torchscript_graph_executor_parallelism_and_compile_remain_unsupported(self):
        self.assertTrue(callable(torch.jit.optimized_execution))
        self.assertIs(torch.jit.is_scripting(), False)
        self.assertIs(torch.jit.is_tracing(), False)
        for name in (
            "CompilationUnit",
            "ScriptFunction",
            "ScriptModule",
            "enable_onednn_fusion",
            "fork",
            "last_executed_optimized_graph",
            "script",
            "set_fusion_strategy",
            "trace",
            "trace_module",
            "wait",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.jit, name))
        for name in (
            "_get_graph_executor_optimize",
            "_set_graph_executor_optimize",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch._C, name))
        self.assertTrue(callable(torch.compile))

    def test_supported_scope_is_documented(self):
        supported_surface = (ROOT / "docs" / "supported-surface.md").read_text(
            encoding="utf-8"
        )
        features = (ROOT / "FEATURES.md").read_text(encoding="utf-8")
        for text in (supported_surface, features):
            for phrase in (
                "torch.jit.optimized_execution",
                "torch.jit.script",
                "torch.jit.trace",
                "torch.jit.fork",
                "torch.jit.wait",
                "graph executor optimization",
            ):
                with self.subTest(phrase=phrase):
                    self.assertIn(phrase, text)

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

events = []

class BoolLike:
    def __bool__(self):
        events.append("entered")
        return True

with torch.jit.optimized_execution(BoolLike()) as entered:
    assert entered is None
assert events == ["entered"]

@torch.jit.optimized_execution(False)
def function(value):
    return value + 1

assert function(2) == 3
assert torch.jit.optimized_execution.__module__ == "torch_rs.jit._fuser"
assert "optimized_execution" not in torch.jit.__all__
assert not hasattr(torch.jit, "script")
assert not hasattr(torch.jit, "trace")
assert not hasattr(torch.jit, "fork")
assert not hasattr(torch.jit, "wait")
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
