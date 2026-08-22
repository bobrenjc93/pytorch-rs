import collections.abc
import copy
import importlib
import inspect
import pickle
import subprocess
import sys
import types
import typing
import unittest

import torch_rs as torch


FUNCTION_DOC = """
    Compiles ``fn`` when it is first called during tracing.

    .. deprecated:: 2.5
        TorchScript is deprecated, please use ``torch.compile`` instead.

    ``torch.jit.script`` has a non-negligible start up time when it is first called due to
    lazy-initializations of many compiler builtins. Therefore you should not use
    it in library code. However, you may want to have parts of your library work
    in tracing even if they use control flow. In these cases, you should use
    ``@torch.jit.script_if_tracing`` to substitute for
    ``torch.jit.script``.

    Args:
        fn: A function to compile.

    Returns:
        If called during tracing, a :class:`ScriptFunction` created by `torch.jit.script` is returned.
        Otherwise, the original function `fn` is returned.
    """


@torch.jit.script_if_tracing
def _picklable_script_if_tracing_function(value):
    return value


class _PicklableMethods:
    @torch.jit.script_if_tracing
    def method(self, value):
        return value + 1


class JitScriptIfTracingTests(unittest.TestCase):
    def test_eager_calls_forward_arguments_results_and_exceptions(self):
        result = object()
        first = object()
        second = object()
        third = object()
        fourth = object()
        fifth = object()
        calls = []

        def function(positional, /, optional=None, *args, keyword, **kwargs):
            calls.append(
                (
                    positional,
                    optional,
                    args,
                    keyword,
                    kwargs,
                    torch.jit.is_tracing(),
                )
            )
            return result

        wrapped = torch.jit.script_if_tracing(function)
        self.assertIsNot(wrapped, function)
        self.assertIs(
            wrapped(first, second, third, keyword=fourth, named=fifth),
            result,
        )
        self.assertEqual(len(calls), 1)
        recorded = calls[0]
        self.assertIs(recorded[0], first)
        self.assertIs(recorded[1], second)
        self.assertEqual(recorded[2], (third,))
        self.assertIs(recorded[2][0], third)
        self.assertIs(recorded[3], fourth)
        self.assertEqual(set(recorded[4]), {"named"})
        self.assertIs(recorded[4]["named"], fifth)
        self.assertIs(recorded[5], False)

        error = RuntimeError("forwarded failure", result)
        error_calls = []

        def raising(*args, **kwargs):
            error_calls.append((args, kwargs))
            raise error

        wrapped_raising = torch.jit.script_if_tracing(raising)
        with self.assertRaises(RuntimeError) as raised:
            wrapped_raising(first, named=second)
        self.assertIs(raised.exception, error)
        self.assertEqual(error_calls, [((first,), {"named": second})])
        self.assertIs(torch.jit.is_tracing(), False)

    def test_wrapper_preserves_metadata_and_sets_pytorch_markers(self):
        sentinel = object()

        def function(
            value: int,
            /,
            scale: float = 2.0,
            *,
            label: str = "value",
        ) -> tuple[int, float, str]:
            """Function documentation."""
            return value, scale, label

        function.custom_attribute = sentinel
        setattr(function, "__original_fn", "stale original")
        setattr(function, "__script_if_tracing_wrapper", False)

        wrapped = torch.jit.script_if_tracing(function)

        self.assertIs(type(wrapped), types.FunctionType)
        self.assertIsNot(wrapped, function)
        self.assertEqual(wrapped.__name__, function.__name__)
        self.assertEqual(wrapped.__qualname__, function.__qualname__)
        self.assertEqual(wrapped.__module__, function.__module__)
        self.assertEqual(wrapped.__doc__, function.__doc__)
        self.assertEqual(wrapped.__annotations__, function.__annotations__)
        self.assertEqual(
            typing.get_type_hints(wrapped), typing.get_type_hints(function)
        )
        self.assertIsNone(wrapped.__defaults__)
        self.assertIsNone(wrapped.__kwdefaults__)
        self.assertIs(wrapped.custom_attribute, sentinel)
        self.assertIs(wrapped.__wrapped__, function)
        self.assertIs(getattr(wrapped, "__original_fn"), function)
        self.assertIs(getattr(wrapped, "__script_if_tracing_wrapper"), True)
        self.assertIs(inspect.unwrap(wrapped), function)
        self.assertEqual(inspect.signature(wrapped), inspect.signature(function))
        self.assertEqual(
            set(wrapped.__dict__),
            {
                "custom_attribute",
                "__wrapped__",
                "__original_fn",
                "__script_if_tracing_wrapper",
            },
        )
        self.assertEqual(wrapped(3, label="scaled"), (3, 2.0, "scaled"))

    def test_methods_and_callable_objects_preserve_python_binding(self):
        class Example:
            @torch.jit.script_if_tracing
            def method(self, value):
                return self, value + 1

            @staticmethod
            @torch.jit.script_if_tracing
            def static_method(value):
                return value + 2

            @classmethod
            @torch.jit.script_if_tracing
            def class_method(cls, value):
                return cls, value + 3

        instance = Example()
        raw_method = Example.__dict__["method"]
        bound_instance, bound_value = instance.method(4)
        self.assertIs(bound_instance, instance)
        self.assertEqual(bound_value, 5)
        self.assertIs(getattr(raw_method, "__script_if_tracing_wrapper"), True)
        self.assertEqual(
            getattr(raw_method, "__original_fn").__qualname__,
            raw_method.__qualname__,
        )
        self.assertEqual(Example.static_method(4), 6)
        self.assertEqual(instance.static_method(4), 6)
        class_owner, class_value = instance.class_method(4)
        self.assertIs(class_owner, Example)
        self.assertEqual(class_value, 7)

        class CallableTarget:
            def __call__(self, value, *, increment=1):
                return value + increment

        target = CallableTarget()
        wrapped_target = torch.jit.script_if_tracing(target)
        self.assertIs(getattr(wrapped_target, "__original_fn"), target)
        self.assertIs(wrapped_target.__wrapped__, target)
        self.assertEqual(inspect.signature(wrapped_target), inspect.signature(target))
        self.assertEqual(wrapped_target(4, increment=5), 9)

        wrapped_len = torch.jit.script_if_tracing(len)
        self.assertEqual(wrapped_len([1, 2, 3]), 3)
        self.assertEqual(wrapped_len.__name__, "len")
        self.assertIs(getattr(wrapped_len, "__original_fn"), len)

    def test_signature_documentation_and_module_ownership(self):
        jit = importlib.import_module("torch_rs.jit")
        trace = importlib.import_module("torch_rs.jit._trace")
        function = jit.script_if_tracing
        internal = trace._script_if_tracing

        self.assertIs(torch.jit, jit)
        self.assertIs(jit._trace, trace)
        self.assertIs(sys.modules["torch_rs.jit._trace"], trace)
        self.assertIsNot(function, internal)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "(fn)")
        self.assertEqual(function.__annotations__, {})
        self.assertEqual(typing.get_type_hints(function), {})
        self.assertEqual(function.__name__, "script_if_tracing")
        self.assertEqual(function.__qualname__, "script_if_tracing")
        self.assertEqual(function.__module__, "torch_rs.jit")
        self.assertIs(inspect.getmodule(function), jit)
        self.assertEqual(
            inspect.cleandoc(function.__doc__), inspect.cleandoc(FUNCTION_DOC)
        )
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

        self.assertIs(type(internal), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(internal)),
            "(fn: collections.abc.Callable[~P, +R]) -> "
            "collections.abc.Callable[~P, +R]",
        )
        self.assertEqual(set(internal.__annotations__), {"fn", "return"})
        parameters, result = typing.get_args(internal.__annotations__["fn"])
        self.assertIs(
            typing.get_origin(internal.__annotations__["fn"]),
            collections.abc.Callable,
        )
        self.assertIs(parameters, trace._P)
        self.assertIs(result, trace._R)
        self.assertEqual(trace._P.__name__, "P")
        self.assertEqual(trace._R.__name__, "R")
        self.assertEqual(internal.__name__, "_script_if_tracing")
        self.assertEqual(internal.__qualname__, "_script_if_tracing")
        self.assertEqual(internal.__module__, "torch_rs.jit._trace")
        self.assertIs(inspect.getmodule(internal), trace)
        self.assertIsNone(internal.__doc__)
        self.assertEqual(internal.__dict__, {})

        def keyword_target():
            return "keyword target"

        keyword_wrapper = function(fn=keyword_target)
        self.assertIs(getattr(keyword_wrapper, "__original_fn"), keyword_target)
        self.assertEqual(keyword_wrapper(), "keyword target")

    def test_exports_copy_and_pickle_use_canonical_modules(self):
        jit = torch.jit
        trace = jit._trace
        function = jit.script_if_tracing
        internal = trace._script_if_tracing

        self.assertEqual(
            jit.__all__,
            [
                "Attribute",
                "annotate",
                "export",
                "ignore",
                "isinstance",
                "onednn_fusion_enabled",
                "script_if_tracing",
                "strict_fusion",
                "unused",
            ],
        )
        self.assertEqual(
            {name for name in vars(jit) if not name.startswith("_")},
            {
                "Attribute",
                "annotate",
                "export",
                "ignore",
                "isinstance",
                "onednn_fusion_enabled",
                "is_scripting",
                "is_tracing",
                "script_if_tracing",
                "strict_fusion",
                "unused",
            },
        )
        self.assertFalse(hasattr(trace, "__all__"))
        self.assertEqual(
            {name for name in vars(trace) if not name.startswith("_")},
            {"is_tracing"},
        )

        package_namespace = {}
        exec("from torch_rs.jit import script_if_tracing", package_namespace)
        self.assertIs(package_namespace["script_if_tracing"], function)

        wildcard_namespace = {}
        exec("from torch_rs.jit import *", wildcard_namespace)
        self.assertEqual(
            {name for name in wildcard_namespace if not name.startswith("__")},
            {
                "Attribute",
                "annotate",
                "export",
                "ignore",
                "isinstance",
                "onednn_fusion_enabled",
                "script_if_tracing",
                "strict_fusion",
                "unused",
            },
        )
        self.assertIs(wildcard_namespace["script_if_tracing"], function)

        trace_namespace = {}
        exec("from torch_rs.jit._trace import _script_if_tracing", trace_namespace)
        self.assertIs(trace_namespace["_script_if_tracing"], internal)
        trace_wildcard_namespace = {}
        exec("from torch_rs.jit._trace import *", trace_wildcard_namespace)
        self.assertNotIn("_script_if_tracing", trace_wildcard_namespace)

        self.assertNotIn("jit", torch.__all__)
        self.assertNotIn("script_if_tracing", torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("jit", top_level_namespace)
        self.assertNotIn("script_if_tracing", top_level_namespace)
        self.assertFalse(hasattr(torch, "script_if_tracing"))

        method = _PicklableMethods.__dict__["method"]
        for value in (
            function,
            internal,
            _picklable_script_if_tracing_function,
            method,
        ):
            with self.subTest(value=value):
                self.assertIs(copy.copy(value), value)
                self.assertIs(copy.deepcopy(value), value)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    payload = pickle.dumps(value, protocol=protocol)
                    self.assertIs(pickle.loads(payload), value)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIn(
                    b"torch_rs.jit",
                    pickle.dumps(function, protocol=protocol),
                )
                self.assertIn(
                    b"torch_rs.jit._trace",
                    pickle.dumps(internal, protocol=protocol),
                )

    def test_invalid_calls_and_targets_match_pytorch_2_13_errors(self):
        function = torch.jit.script_if_tracing
        cases = (
            (
                lambda: function(),
                "script_if_tracing() missing 1 required positional argument: 'fn'",
            ),
            (
                lambda: function(lambda: None, lambda: None),
                "script_if_tracing() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: function(function=lambda: None),
                "script_if_tracing() got an unexpected keyword argument 'function'",
            ),
            (
                lambda: function(lambda: None, fn=lambda: None),
                "script_if_tracing() got multiple values for argument 'fn'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

        invalid_targets = (
            (None, "'NoneType' object is not callable"),
            (1, "'int' object is not callable"),
            (property(), "'property' object is not callable"),
        )
        for target, message in invalid_targets:
            with self.subTest(target=target):
                wrapped = function(target)
                self.assertIs(wrapped.__wrapped__, target)
                self.assertIs(getattr(wrapped, "__original_fn"), target)
                self.assertIs(
                    getattr(wrapped, "__script_if_tracing_wrapper"), True
                )
                with self.assertRaises(TypeError) as raised:
                    wrapped()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_scripting_tracing_and_compilation_remain_unsupported(self):
        self.assertTrue(callable(torch.jit.script_if_tracing))
        self.assertIs(torch.jit.is_tracing(), False)
        for name in (
            "CompilationUnit",
            "ScriptFunction",
            "ScriptModule",
            "script",
            "script_method",
            "trace",
            "trace_module",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.jit, name))
        for name in ("script", "trace", "trace_module"):
            with self.subTest(trace_name=name):
                self.assertFalse(hasattr(torch.jit._trace, name))
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

def function(value, *, increment=1):
    return value + increment

wrapped = torch.jit.script_if_tracing(function)
assert wrapped is not function
assert wrapped(2, increment=3) == 5
assert wrapped.__wrapped__ is function
assert wrapped.__original_fn is function
assert wrapped.__script_if_tracing_wrapper is True
assert torch.jit.is_tracing() is False

class Example:
    @torch.jit.script_if_tracing
    def method(self, value):
        return value + 1

assert Example().method(2) == 3
assert not hasattr(torch.jit, "script")
assert not hasattr(torch.jit, "trace")
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
