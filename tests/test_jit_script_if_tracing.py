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


def _picklable_script_if_tracing_function(value):
    return value + 1


_picklable_script_if_tracing_function = torch.jit.script_if_tracing(
    _picklable_script_if_tracing_function
)


class JitScriptIfTracingTests(unittest.TestCase):
    def test_eager_wrapper_forwards_calls_results_and_exceptions(self):
        default = object()
        result = object()
        raised = ValueError("original failure")
        calls = []

        def function(value, other=default, *, option=default):
            calls.append((value, other, option, torch.jit.is_tracing()))
            if value == "raise":
                raise raised
            return result

        wrapper = torch.jit.script_if_tracing(function)

        self.assertIsNot(wrapper, function)
        self.assertIs(wrapper("value"), result)
        self.assertIs(wrapper(value="keyword", option=None), result)
        self.assertEqual(
            calls,
            [
                ("value", default, default, False),
                ("keyword", default, None, False),
            ],
        )
        with self.assertRaises(ValueError) as caught:
            wrapper("raise")
        self.assertIs(caught.exception, raised)
        self.assertEqual(calls[-1], ("raise", default, default, False))

    def test_wrapper_preserves_function_metadata_and_sets_markers(self):
        sentinel = object()

        def function(
            value: int, offset: int = 2, *, option: object = sentinel
        ) -> tuple[int, int, object]:
            """Original function documentation."""
            return value, offset, option

        function.custom_attribute = sentinel
        original_metadata = (
            function.__name__,
            function.__qualname__,
            function.__module__,
            function.__doc__,
            function.__annotations__.copy(),
        )

        wrapper = torch.jit.script_if_tracing(function)

        self.assertEqual(
            (
                wrapper.__name__,
                wrapper.__qualname__,
                wrapper.__module__,
                wrapper.__doc__,
                wrapper.__annotations__,
            ),
            original_metadata,
        )
        self.assertEqual(inspect.signature(wrapper), inspect.signature(function))
        self.assertIs(wrapper.custom_attribute, sentinel)
        self.assertIs(wrapper.__wrapped__, function)
        self.assertIs(getattr(wrapper, "__original_fn"), function)
        self.assertIs(getattr(wrapper, "__script_if_tracing_wrapper"), True)
        self.assertIs(inspect.unwrap(wrapper), function)
        self.assertIsNone(wrapper.__defaults__)
        self.assertIsNone(wrapper.__kwdefaults__)
        self.assertEqual(wrapper(3, option="set"), (3, 2, "set"))

        if hasattr(function, "__type_params__"):
            self.assertEqual(wrapper.__type_params__, function.__type_params__)

    def test_existing_marker_attributes_are_overwritten_and_nesting_is_stable(self):
        def function(value):
            return value + 1

        function.__wrapped__ = "old wrapped value"
        setattr(function, "__original_fn", "old original value")
        setattr(function, "__script_if_tracing_wrapper", False)
        function.custom_attribute = "preserved"

        first = torch.jit.script_if_tracing(function)
        second = torch.jit.script_if_tracing(first)

        self.assertIs(first.__wrapped__, function)
        self.assertIs(getattr(first, "__original_fn"), function)
        self.assertIs(getattr(first, "__script_if_tracing_wrapper"), True)
        self.assertEqual(first.custom_attribute, "preserved")
        self.assertIs(second.__wrapped__, first)
        self.assertIs(getattr(second, "__original_fn"), first)
        self.assertIs(getattr(second, "__script_if_tracing_wrapper"), True)
        self.assertEqual(second(4), 5)

    def test_wrapped_methods_preserve_descriptor_binding(self):
        class Example:
            @torch.jit.script_if_tracing
            def method(self, value=1):
                return self, value

            @staticmethod
            @torch.jit.script_if_tracing
            def static_method(value=2):
                return value

            @classmethod
            @torch.jit.script_if_tracing
            def class_method(cls, value=3):
                return cls, value

        instance = Example()
        raw_method = Example.__dict__["method"]
        raw_static_method = Example.__dict__["static_method"].__func__
        raw_class_method = Example.__dict__["class_method"].__func__

        self.assertIs(instance.method.__self__, instance)
        self.assertIs(instance.method.__func__, raw_method)
        self.assertIs(instance.method()[0], instance)
        self.assertEqual(instance.method()[1], 1)
        self.assertEqual(Example.static_method(), 2)
        self.assertEqual(instance.static_method(5), 5)
        self.assertIs(Example.class_method()[0], Example)
        self.assertEqual(instance.class_method(6)[1], 6)
        for wrapper in (raw_method, raw_static_method, raw_class_method):
            self.assertIs(getattr(wrapper, "__script_if_tracing_wrapper"), True)
            self.assertIs(wrapper.__wrapped__, getattr(wrapper, "__original_fn"))

        bound_wrapper = torch.jit.script_if_tracing(instance.method)
        self.assertEqual(bound_wrapper(7), (instance, 7))
        self.assertEqual(inspect.signature(bound_wrapper), inspect.signature(instance.method))

    def test_callable_objects_and_builtins_are_forwarded(self):
        class CallableTarget:
            def __init__(self):
                self.custom_attribute = "copied"

            def __call__(self, value, *, scale=2):
                return value * scale

        target = CallableTarget()
        wrapped_target = torch.jit.script_if_tracing(target)
        wrapped_len = torch.jit.script_if_tracing(len)

        self.assertEqual(wrapped_target(4, scale=3), 12)
        self.assertIs(getattr(wrapped_target, "__original_fn"), target)
        self.assertEqual(wrapped_target.custom_attribute, "copied")
        self.assertEqual(wrapped_len([1, 2, 3]), 3)
        self.assertEqual(wrapped_len.__name__, "len")
        self.assertIs(getattr(wrapped_len, "__original_fn"), len)

    def test_signature_documentation_and_ownership(self):
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
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

        self.assertEqual(helper.__name__, "_script_if_tracing")
        self.assertEqual(helper.__qualname__, "_script_if_tracing")
        self.assertEqual(helper.__module__, "torch_rs.jit._trace")
        self.assertIs(inspect.getmodule(helper), trace)
        self.assertEqual(
            str(inspect.signature(helper)),
            "(fn: collections.abc.Callable[~P, +R]) -> collections.abc.Callable[~P, +R]",
        )

    def test_exports_copy_and_pickle_use_canonical_modules(self):
        jit = torch.jit
        trace = jit._trace
        function = jit.script_if_tracing
        wrapped = _picklable_script_if_tracing_function

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

        helper_namespace = {}
        exec("from torch_rs.jit._trace import _script_if_tracing", helper_namespace)
        self.assertIs(helper_namespace["_script_if_tracing"], trace._script_if_tracing)

        jit_wildcard_namespace = {}
        exec("from torch_rs.jit import *", jit_wildcard_namespace)
        self.assertEqual(
            {
                name
                for name in jit_wildcard_namespace
                if not name.startswith("__")
            },
            {"annotate", "export", "ignore", "script_if_tracing", "unused"},
        )
        self.assertIs(jit_wildcard_namespace["script_if_tracing"], function)

        trace_wildcard_namespace = {}
        exec("from torch_rs.jit._trace import *", trace_wildcard_namespace)
        self.assertEqual(
            {
                name
                for name in trace_wildcard_namespace
                if not name.startswith("__")
            },
            {"is_tracing"},
        )

        self.assertNotIn("script_if_tracing", torch.__all__)
        self.assertFalse(hasattr(torch, "script_if_tracing"))
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("script_if_tracing", top_level_namespace)

        for value in (function, trace._script_if_tracing, wrapped):
            with self.subTest(value=value):
                self.assertIs(copy.copy(value), value)
                self.assertIs(copy.deepcopy(value), value)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    payload = pickle.dumps(value, protocol=protocol)
                    self.assertIs(pickle.loads(payload), value)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            self.assertIn(
                b"torch_rs.jit",
                pickle.dumps(function, protocol=protocol),
            )

    def test_invalid_calls_and_targets_match_eager_wrapper_behavior(self):
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
                lambda: function(target=lambda: None),
                "script_if_tracing() got an unexpected keyword argument 'target'",
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
            ("invalid", "'str' object is not callable"),
            ([], "'list' object is not callable"),
            (property(), "'property' object is not callable"),
        )
        for target, message in invalid_targets:
            with self.subTest(target=target):
                wrapper = function(target)
                self.assertIs(wrapper.__wrapped__, target)
                self.assertIs(getattr(wrapper, "__original_fn"), target)
                self.assertIs(getattr(wrapper, "__script_if_tracing_wrapper"), True)
                with self.assertRaises(TypeError) as raised:
                    wrapper()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_scripting_tracing_and_compilation_remain_unsupported(self):
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
        self.assertFalse(hasattr(torch.jit._trace, "trace"))
        self.assertFalse(hasattr(torch.jit._trace, "trace_module"))
        self.assertFalse(hasattr(torch, "compile"))

    def test_importing_and_using_the_decorator_does_not_import_pytorch(self):
        script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

@torch.jit.script_if_tracing
def function(value, *, offset=1):
    assert torch.jit.is_tracing() is False
    return value + offset

assert function(2, offset=3) == 5
assert function.__original_fn(2, offset=4) == 6
assert function.__wrapped__ is function.__original_fn
assert function.__script_if_tracing_wrapper is True
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
