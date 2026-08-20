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


class JitScriptIfTracingTests(unittest.TestCase):
    def test_eager_wrapper_forwards_calls_results_and_exceptions_exactly(self):
        default = object()
        positional = object()
        optional = object()
        extra = object()
        keyword = object()
        named = object()
        result = object()
        calls = []

        def function(
            first,
            /,
            second=default,
            *args,
            required,
            optional_keyword=default,
            **kwargs,
        ):
            calls.append(
                (first, second, args, required, optional_keyword, kwargs)
            )
            return result

        wrapped = torch.jit.script_if_tracing(function)
        self.assertIsNot(wrapped, function)
        self.assertIs(
            wrapped(
                positional,
                optional,
                extra,
                required=keyword,
                named=named,
            ),
            result,
        )
        self.assertEqual(len(calls), 1)
        first, second, extras, required, optional_keyword, kwargs = calls[0]
        self.assertIs(first, positional)
        self.assertIs(second, optional)
        self.assertEqual(extras, (extra,))
        self.assertIs(extras[0], extra)
        self.assertIs(required, keyword)
        self.assertIs(optional_keyword, default)
        self.assertEqual(set(kwargs), {"named"})
        self.assertIs(kwargs["named"], named)

        expected_error = RuntimeError("same exception")

        def raises(*args, **kwargs):
            self.assertEqual(args, (positional,))
            self.assertEqual(kwargs, {"keyword": keyword})
            raise expected_error

        wrapped_raises = torch.jit.script_if_tracing(raises)
        with self.assertRaises(RuntimeError) as raised:
            wrapped_raises(positional, keyword=keyword)
        self.assertIs(raised.exception, expected_error)
        self.assertEqual(raised.exception.args, ("same exception",))
        self.assertIs(torch.jit.is_tracing(), False)

    def test_wrapper_preserves_metadata_and_sets_pytorch_markers(self):
        custom_value = {"nested": []}
        previous_original = object()

        def function(value: int = 3, *, enabled: bool = True) -> tuple:
            """Function documentation."""
            return value, enabled

        function.custom_attribute = custom_value
        setattr(function, "__original_fn", previous_original)
        setattr(function, "__script_if_tracing_wrapper", False)

        wrapped = torch.jit.script_if_tracing(function)

        self.assertIs(type(wrapped), types.FunctionType)
        self.assertEqual(wrapped.__name__, function.__name__)
        self.assertEqual(wrapped.__qualname__, function.__qualname__)
        self.assertEqual(wrapped.__module__, function.__module__)
        self.assertEqual(wrapped.__doc__, function.__doc__)
        self.assertEqual(wrapped.__annotations__, function.__annotations__)
        self.assertIs(wrapped.custom_attribute, custom_value)
        self.assertIs(wrapped.__wrapped__, function)
        self.assertIs(getattr(wrapped, "__original_fn"), function)
        self.assertIs(getattr(wrapped, "__script_if_tracing_wrapper"), True)
        self.assertIs(getattr(function, "__original_fn"), previous_original)
        self.assertIs(getattr(function, "__script_if_tracing_wrapper"), False)
        self.assertIs(inspect.unwrap(wrapped), function)
        self.assertEqual(inspect.signature(wrapped), inspect.signature(function))
        self.assertEqual(
            str(inspect.signature(wrapped, follow_wrapped=False)),
            "(*args, **kwargs) -> tuple",
        )
        self.assertIsNone(wrapped.__defaults__)
        self.assertIsNone(wrapped.__kwdefaults__)
        self.assertEqual(wrapped(8, enabled=False), (8, False))

    def test_functions_methods_and_descriptors_bind_like_python_functions(self):
        class Example:
            @torch.jit.script_if_tracing
            def method(self, value, *, offset=1):
                return self, value + offset

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
        bound_method = instance.method
        self.assertIs(bound_method.__self__, instance)
        self.assertIs(bound_method.__func__, raw_method)
        self.assertIs(getattr(raw_method, "__script_if_tracing_wrapper"), True)
        self.assertEqual(getattr(raw_method, "__original_fn").__name__, "method")
        owner, value = bound_method(4, offset=5)
        self.assertIs(owner, instance)
        self.assertEqual(value, 9)

        self.assertEqual(Example.static_method(4), 6)
        owner, value = Example.class_method(4)
        self.assertIs(owner, Example)
        self.assertEqual(value, 7)

        class CallableTarget:
            def __init__(self):
                self.custom_attribute = object()

            def __call__(self, value):
                return value

        target = CallableTarget()
        wrapped = torch.jit.script_if_tracing(target)
        sentinel = object()
        self.assertIs(wrapped(sentinel), sentinel)
        self.assertIs(getattr(wrapped, "__original_fn"), target)
        self.assertIs(wrapped.__wrapped__, target)
        self.assertIs(wrapped.custom_attribute, target.custom_attribute)

    def test_signature_documentation_and_module_ownership(self):
        jit = importlib.import_module("torch_rs.jit")
        trace = importlib.import_module("torch_rs.jit._trace")
        function = jit.script_if_tracing
        helper = trace._script_if_tracing

        self.assertIs(torch.jit, jit)
        self.assertIs(jit._trace, trace)
        self.assertIs(jit._script_if_tracing, helper)
        self.assertIsNot(function, helper)
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

        self.assertIs(type(helper), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(helper)),
            "(fn: collections.abc.Callable[~P, +R]) -> "
            "collections.abc.Callable[~P, +R]",
        )
        self.assertEqual(set(helper.__annotations__), {"fn", "return"})
        self.assertIs(
            typing.get_origin(helper.__annotations__["fn"]),
            collections.abc.Callable,
        )
        self.assertEqual(
            helper.__annotations__["fn"], helper.__annotations__["return"]
        )
        parameters, result = typing.get_args(helper.__annotations__["fn"])
        self.assertIs(parameters, trace._P)
        self.assertIs(result, trace._R)
        self.assertEqual(trace._P.__name__, "P")
        self.assertEqual(trace._R.__name__, "R")
        self.assertEqual(helper.__name__, "_script_if_tracing")
        self.assertEqual(helper.__qualname__, "_script_if_tracing")
        self.assertEqual(helper.__module__, "torch_rs.jit._trace")
        self.assertIs(inspect.getmodule(helper), trace)
        self.assertIsNone(helper.__doc__)
        self.assertIsNone(helper.__defaults__)
        self.assertIsNone(helper.__kwdefaults__)
        self.assertEqual(helper.__dict__, {})

    def test_exports_copy_and_pickle_match_pytorch_ownership(self):
        jit = torch.jit
        trace = jit._trace
        function = jit.script_if_tracing
        helper = trace._script_if_tracing

        self.assertEqual(
            jit.__all__,
            ["annotate", "export", "ignore", "script_if_tracing", "unused"],
        )
        self.assertEqual(
            {name for name in vars(jit) if not name.startswith("_")},
            {
                "annotate",
                "export",
                "ignore",
                "is_scripting",
                "is_tracing",
                "script_if_tracing",
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
        private_namespace = {}
        exec(
            "from torch_rs.jit._trace import _script_if_tracing",
            private_namespace,
        )
        self.assertIs(private_namespace["_script_if_tracing"], helper)

        jit_namespace = {}
        exec("from torch_rs.jit import *", jit_namespace)
        self.assertEqual(
            {name for name in jit_namespace if not name.startswith("__")},
            {"annotate", "export", "ignore", "script_if_tracing", "unused"},
        )
        self.assertIs(jit_namespace["script_if_tracing"], function)

        trace_namespace = {}
        exec("from torch_rs.jit._trace import *", trace_namespace)
        self.assertEqual(
            {name for name in trace_namespace if not name.startswith("__")},
            {"is_tracing"},
        )
        self.assertNotIn("_script_if_tracing", trace_namespace)

        self.assertNotIn("jit", torch.__all__)
        self.assertNotIn("script_if_tracing", torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("jit", top_level_namespace)
        self.assertNotIn("script_if_tracing", top_level_namespace)
        self.assertFalse(hasattr(torch, "script_if_tracing"))

        for value in (function, helper, _picklable_script_if_tracing_function):
            with self.subTest(value=value):
                self.assertIs(copy.copy(value), value)
                self.assertIs(copy.deepcopy(value), value)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    self.assertIs(
                        pickle.loads(pickle.dumps(value, protocol=protocol)), value
                    )

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIn(
                    b"torch_rs.jit", pickle.dumps(function, protocol=protocol)
                )
                self.assertIn(
                    b"torch_rs.jit._trace", pickle.dumps(helper, protocol=protocol)
                )

    def test_invalid_calls_match_pytorch_2_13_eager_behavior(self):
        function = torch.jit.script_if_tracing
        call_cases = (
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
        for call, message in call_cases:
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
                wrapped = function(fn=target)
                self.assertIs(wrapped.__wrapped__, target)
                self.assertIs(getattr(wrapped, "__original_fn"), target)
                self.assertIs(
                    getattr(wrapped, "__script_if_tracing_wrapper"), True
                )
                with self.assertRaises(TypeError) as raised:
                    wrapped()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

        wrapped_len = function(len)
        self.assertEqual(wrapped_len([1, 2, 3]), 3)
        with self.assertRaises(TypeError) as raised:
            wrapped_len()
        self.assertEqual(str(raised.exception), "len() takes exactly one argument (0 given)")

    def test_script_and_trace_remain_unsupported(self):
        self.assertTrue(callable(torch.jit.script_if_tracing))
        self.assertIs(torch.jit.is_tracing(), False)
        self.assertIs(torch.jit.is_scripting(), False)
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
        self.assertFalse(hasattr(torch.jit._trace, "script"))
        self.assertFalse(hasattr(torch.jit._trace, "trace"))
        self.assertFalse(hasattr(torch.jit._trace, "trace_module"))
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

def function(value):
    return value

wrapped = torch.jit.script_if_tracing(function)
sentinel = object()
assert wrapped(sentinel) is sentinel
assert wrapped is not function
assert wrapped.__wrapped__ is function
assert wrapped.__original_fn is function
assert wrapped.__script_if_tracing_wrapper is True
assert torch.jit.is_tracing() is False
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
