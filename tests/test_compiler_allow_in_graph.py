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
    Tells the compiler frontend (Dynamo) to skip symbolic introspection of the function
    and instead directly write it to the graph when encountered.

    If you are using :func:`torch.compile` (with backend="inductor" (the default)), or
    :func:`torch.export.export`, and trying to black-box a Python function throughout
    all tracing, do not use this API.
    Instead, please create a custom operator (see `PyTorch Custom Operators Landing Page
    <https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html>`_)

    .. warning::

        If you're a typical torch.compile user (e.g. you're applying torch.compile to
        a model to make it run faster), you probably don't want to use this function.
        :func:`allow_in_graph` is a footgun because it skips the compiler frontend
        (Dynamo) that is responsible for doing safety checks (graph breaks, handling
        closures, etc). Incorrect usage will lead to difficult-to-debug silent
        incorrectness issues.

    Given a Python function with no allow_in_graph decorator, regular execution
    of torch.compile traces through the function. :func:`allow_in_graph` changes
    it so that the frontend does not trace inside the function, but the compiler
    backend still traces through it. Compare this to custom operators, which
    treats a function as a black box throughout the torch.compile stack. The following
    table compares these mechanisms.

    +------------------------+-----------------------+--------------------------------+
    | Mechanism              | Frontend (Dynamo)     | Backend (AOTAutograd+Inductor) |
    +========================+=======================+================================+
    | no decorator           | trace inside          | trace inside                   |
    +------------------------+-----------------------+--------------------------------+
    | allow_in_graph         | opaque callable       | trace inside                   |
    +------------------------+-----------------------+--------------------------------+
    | custom op              | opaque callable       | opaque callable                |
    +------------------------+-----------------------+--------------------------------+

    One common use case for :func:`allow_in_graph()` is as an escape hatch for the compiler
    frontend: if you know the function works w.r.t. to the downstream components of the
    compilation stack (AOTAutograd and Inductor) but there is a Dynamo bug that prevents it from
    symbolically introspecting the function properly (or if your code is in C/C++ and
    therefore cannot be introspected with Dynamo), then one can decorate said function
    with :func:`allow_in_graph` to bypass Dynamo.

    We require that ``fn`` adhere to the following restrictions. Failure to adhere
    results in undefined behavior:

    - The inputs to ``fn`` must be Proxy-able types in the FX graph. Valid types include:
      Tensor/int/bool/float/None/List[Tensor?]/List[int?]/List[float?]
      Tuple[Tensor?, ...]/Tuple[int?, ...]/Tuple[float?, ...]/torch.dtype/torch.device
    - The outputs to ``fn`` must be Proxy-able types in the FX graph (see previous bullet)
    - all Tensors used inside of ``fn`` must be passed directly as inputs to ``fn``
      (as opposed to being captured variables).

    Args:
        fn: A callable representing the function to be included in the graph.
            If ``fn`` is a list or tuple of callables it recursively applies
            :func:`allow_in_graph()` to each function and returns a new list or
            tuple containing the modified functions.

    Example::

        torch.compiler.allow_in_graph(my_custom_function)


        @torch.compile(...)
        def fn(x):
            x = torch.add(x, 1)
            x = my_custom_function(x)
            x = torch.add(x, 1)
            return x


        fn(...)

    Will capture a single graph containing ``my_custom_function()``.

    """


COMPILER_EXPORTS = [
    "assume_constant_result",
    "reset",
    "allow_in_graph",
    "list_backends",
    "disable",
    "set_default_backend",
    "get_default_backend",
    "set_enable_guard_collectives",
    "is_compiling",
    "is_dynamo_compiling",
    "is_exporting",
    "keep_portable_guards_unsafe",
    "skip_guard_on_inbuilt_nn_modules_unsafe",
    "skip_guard_on_all_nn_modules_unsafe",
    "keep_tensor_guards_unsafe",
    "skip_guard_on_globals_unsafe",
    "skip_all_guards_unsafe",
]


@torch.compiler.allow_in_graph
def _picklable_allowed_function(value, *, increment=1):
    return value + increment


class _Callable:
    def __init__(self):
        self.calls = []

    def __call__(self, value):
        self.calls.append(value)
        return value


class CompilerAllowInGraphTests(unittest.TestCase):
    def test_decorator_returns_original_function_and_preserves_eager_calls(self):
        calls = []

        @torch.compiler.allow_in_graph
        def calculate(value, *, scale=1):
            """Calculate eagerly."""
            calls.append((value, scale))
            return value * scale + len(calls)

        self.assertEqual(calculate(3, scale=2), 7)
        self.assertEqual(calculate(3, scale=2), 8)
        self.assertEqual(calls, [(3, 2), (3, 2)])
        self.assertEqual(str(inspect.signature(calculate)), "(value, *, scale=1)")
        self.assertEqual(calculate.__name__, "calculate")
        self.assertIn("<locals>.calculate", calculate.__qualname__)
        self.assertEqual(calculate.__module__, __name__)
        self.assertEqual(calculate.__doc__, "Calculate eagerly.")
        self.assertEqual(calculate.__dict__, {})
        self.assertFalse(hasattr(calculate, "__wrapped__"))

    def test_function_lambda_builtin_and_callable_object_inputs_are_not_wrapped(self):
        def function(value):
            return value + 1

        lambda_function = lambda value: value + 2
        callable_object = _Callable()

        function.custom_metadata = ["preserved"]
        lambda_function.custom_metadata = {"preserved": True}

        cases = (
            (function, (4,), 5),
            (lambda_function, (4,), 6),
            (len, ([1, 2, 3],), 3),
            (callable_object, ("value",), "value"),
        )
        for target, args, expected in cases:
            with self.subTest(target=target):
                before_dict = (
                    dict(target.__dict__) if hasattr(target, "__dict__") else None
                )
                result = torch.compiler.allow_in_graph(target)
                after_dict = (
                    dict(target.__dict__) if hasattr(target, "__dict__") else None
                )

                self.assertIs(result, target)
                self.assertEqual(after_dict, before_dict)
                self.assertEqual(result(*args), expected)
                self.assertFalse(hasattr(result, "_dynamo_marked_constant"))
                self.assertFalse(hasattr(result, "_torchdynamo_disable"))

        self.assertEqual(callable_object.calls, ["value"])

    def test_list_and_tuple_inputs_return_new_lists_of_original_callables(self):
        def first():
            return "first"

        def second():
            return "second"

        for container in ([first, second], (first, second)):
            with self.subTest(container_type=type(container)):
                result = torch.compiler.allow_in_graph(container)
                self.assertIs(type(result), list)
                self.assertIsNot(result, container)
                self.assertEqual(len(result), 2)
                self.assertIs(result[0], first)
                self.assertIs(result[1], second)
                self.assertEqual(
                    [function() for function in result],
                    ["first", "second"],
                )

        empty = torch.compiler.allow_in_graph(())
        self.assertIs(type(empty), list)
        self.assertEqual(empty, [])

    def test_noncallable_targets_raise_pytorch_2_13_assertion(self):
        for target in (None, 1, "value", object(), [lambda: None, 1], (1,)):
            with self.subTest(target=target):
                with self.assertRaises(AssertionError) as raised:
                    torch.compiler.allow_in_graph(target)
                self.assertEqual(
                    str(raised.exception),
                    "allow_in_graph expects a callable",
                )
                self.assertEqual(
                    raised.exception.args,
                    ("allow_in_graph expects a callable",),
                )

    def test_signature_annotations_documentation_and_module_identity(self):
        compiler = importlib.import_module("torch_rs.compiler")
        function = compiler.allow_in_graph

        self.assertIs(torch.compiler, compiler)
        self.assertIs(sys.modules["torch_rs.compiler"], compiler)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "(fn)")
        self.assertEqual(function.__annotations__, {})
        self.assertEqual(typing.get_type_hints(function), {})
        self.assertEqual(function.__name__, "allow_in_graph")
        self.assertEqual(function.__qualname__, "allow_in_graph")
        self.assertEqual(function.__module__, "torch_rs.compiler")
        self.assertIs(inspect.getmodule(function), compiler)
        self.assertEqual(
            inspect.cleandoc(function.__doc__), inspect.cleandoc(FUNCTION_DOC)
        )
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_exports_copy_pickle_reload_and_wildcard_imports(self):
        compiler = torch.compiler
        function = compiler.allow_in_graph

        self.assertEqual(compiler.__all__, COMPILER_EXPORTS)
        compiler_namespace = {}
        exec("from torch_rs.compiler import *", compiler_namespace)
        self.assertEqual(
            {name for name in compiler_namespace if not name.startswith("__")},
            set(COMPILER_EXPORTS),
        )
        for name in COMPILER_EXPORTS:
            self.assertIs(compiler_namespace[name], getattr(compiler, name))

        self.assertNotIn("compiler", torch.__all__)
        self.assertNotIn("allow_in_graph", torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("compiler", top_level_namespace)
        self.assertNotIn("allow_in_graph", top_level_namespace)

        for picklable in (function, _picklable_allowed_function):
            self.assertIs(copy.copy(picklable), picklable)
            self.assertIs(copy.deepcopy(picklable), picklable)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(picklable=picklable.__name__, protocol=protocol):
                    payload = pickle.dumps(picklable, protocol=protocol)
                    if picklable is function:
                        self.assertIn(b"torch_rs.compiler", payload)
                    self.assertIs(pickle.loads(payload), picklable)

        self.assertEqual(_picklable_allowed_function(4, increment=3), 7)

        old_function = function
        old_exports = compiler.__all__
        reloaded = importlib.reload(compiler)
        new_function = reloaded.allow_in_graph

        self.assertIs(reloaded, compiler)
        self.assertIs(torch.compiler, compiler)
        self.assertIs(sys.modules["torch_rs.compiler"], compiler)
        self.assertIsNot(new_function, old_function)
        self.assertIsNot(compiler.__all__, old_exports)
        self.assertEqual(compiler.__all__, COMPILER_EXPORTS)
        self.assertIs(new_function(len), len)
        self.assertIs(copy.copy(old_function), old_function)
        self.assertIs(copy.deepcopy(old_function), old_function)
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(old_function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            self.assertIs(
                pickle.loads(pickle.dumps(new_function, protocol)),
                new_function,
            )

    def test_call_shape_errors_match_pytorch_2_13(self):
        marker = torch.compiler.allow_in_graph
        function = lambda: None
        cases = (
            (
                lambda: marker(),
                "allow_in_graph() missing 1 required positional argument: 'fn'",
            ),
            (
                lambda: marker(function, function),
                "allow_in_graph() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: marker(function, fn=function),
                "allow_in_graph() got multiple values for argument 'fn'",
            ),
            (
                lambda: marker(function, extra=True),
                "allow_in_graph() got an unexpected keyword argument 'extra'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_marker_does_not_enable_compiler_or_dynamo_execution_state(self):
        @torch.compiler.allow_in_graph
        def function():
            return (
                torch.compiler.is_compiling(),
                torch.compiler.is_dynamo_compiling(),
                torch.compiler.is_exporting(),
            )

        self.assertEqual(function(), (False, False, False))
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch, "export"))
        self.assertFalse(hasattr(torch.compiler, "compile"))
        self.assertFalse(hasattr(torch.compiler, "register_backend"))
        self.assertFalse(hasattr(torch.compiler, "substitute_in_graph"))
        self.assertFalse(hasattr(torch.compiler, "cudagraph_mark_step_begin"))

    def test_import_and_marking_do_not_import_pytorch_or_dynamo_registry(self):
        script = r"""
import sys

class RejectCompilerImports:
    def find_spec(self, fullname, path=None, target=None):
        if (
            fullname == "torch"
            or fullname.startswith("torch.")
            or fullname.startswith("torch_rs.compiler.backends")
            or fullname == "torch_rs.compiler.registry"
        ):
            raise RuntimeError(f"compiler import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectCompilerImports())
import torch_rs as torch

calls = []

@torch.compiler.allow_in_graph
def function(value):
    calls.append(value)
    return value + 1

assert torch.compiler.allow_in_graph(len) is len
assert function(1) == 2
assert function(2) == 3
assert calls == [1, 2]
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
assert not any(
    name.startswith("torch_rs.compiler.backends") or name == "torch_rs.compiler.registry"
    for name in sys.modules
)
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
