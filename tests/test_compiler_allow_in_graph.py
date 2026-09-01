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


class _CallableObject:
    def __init__(self):
        self.existing = "preserved"

    def __call__(self, value):
        return value + 2


class _SlotCallable:
    __slots__ = ()

    def __call__(self, value):
        return value + 3


class CompilerAllowInGraphTests(unittest.TestCase):
    def test_decorator_returns_original_function_and_preserves_eager_calls(self):
        calls = []

        @torch.compiler.allow_in_graph
        def calculate(value, *, scale=1):
            calls.append((value, scale))
            return value * scale + len(calls)

        self.assertEqual(calculate(3, scale=2), 7)
        self.assertEqual(calculate(3, scale=2), 8)
        self.assertEqual(calls, [(3, 2), (3, 2)])
        self.assertEqual(str(inspect.signature(calculate)), "(value, *, scale=1)")
        self.assertEqual(calculate.__name__, "calculate")
        self.assertIn("<locals>.calculate", calculate.__qualname__)
        self.assertEqual(calculate.__module__, __name__)
        self.assertFalse(hasattr(calculate, "__wrapped__"))
        self.assertFalse(hasattr(calculate, "_dynamo_marked_constant"))
        self.assertFalse(hasattr(calculate, "_torchdynamo_disable"))

    def test_function_lambda_builtin_and_callable_objects_return_by_identity(self):
        def function(value):
            return value + 1

        anonymous = lambda value: value + 2
        callable_object = _CallableObject()
        slot_callable = _SlotCallable()

        cases = (
            (function, (4,), 5),
            (anonymous, (4,), 6),
            (len, ([1, 2, 3],), 3),
            (callable_object, (4,), 6),
            (slot_callable, (4,), 7),
        )
        for target, args, expected in cases:
            with self.subTest(target=target):
                before_dict = (
                    dict(target.__dict__) if hasattr(target, "__dict__") else None
                )
                result = torch.compiler.allow_in_graph(target)

                self.assertIs(result, target)
                self.assertEqual(result(*args), expected)
                if before_dict is not None:
                    self.assertEqual(target.__dict__, before_dict)

    def test_list_and_tuple_inputs_return_new_lists_of_original_callables(self):
        def first():
            return "first"

        second = lambda: "second"
        original_list = [first, second, len]
        original_tuple = (first, second)

        from_list = torch.compiler.allow_in_graph(original_list)
        from_tuple = torch.compiler.allow_in_graph(original_tuple)

        self.assertEqual(from_list, original_list)
        self.assertIsNot(from_list, original_list)
        self.assertIs(from_list[0], first)
        self.assertIs(from_list[1], second)
        self.assertIs(from_list[2], len)
        self.assertEqual(from_tuple, [first, second])
        self.assertIs(type(from_tuple), list)
        self.assertEqual([callable_() for callable_ in from_tuple], ["first", "second"])

    def test_noncallables_raise_pytorch_2_13_assertion(self):
        for target in (None, 1, object(), types.SimpleNamespace(), "text", [1]):
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

    def test_exports_wildcard_copy_pickle_and_reload_use_canonical_objects(self):
        compiler = torch.compiler
        function = compiler.allow_in_graph

        self.assertEqual(compiler.__all__, COMPILER_EXPORTS)
        from torch_rs.compiler import allow_in_graph

        self.assertIs(allow_in_graph, function)

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
        self.assertFalse(hasattr(torch, "allow_in_graph"))
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("compiler", top_level_namespace)
        self.assertNotIn("allow_in_graph", top_level_namespace)

        for copied_function in (function, _picklable_allowed_function):
            self.assertIs(copy.copy(copied_function), copied_function)
            self.assertIs(copy.deepcopy(copied_function), copied_function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(
                    function=copied_function.__name__, protocol=protocol
                ):
                    self.assertIs(
                        pickle.loads(pickle.dumps(copied_function, protocol)),
                        copied_function,
                    )

        self.assertEqual(_picklable_allowed_function(4, increment=3), 7)
        self.assertFalse(hasattr(_picklable_allowed_function, "__wrapped__"))
        self.assertFalse(
            hasattr(_picklable_allowed_function, "_dynamo_marked_constant")
        )
        self.assertFalse(hasattr(_picklable_allowed_function, "_torchdynamo_disable"))

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

    def test_marker_does_not_enable_compiler_execution_or_state(self):
        compiler = torch.compiler
        original_backend = compiler.get_default_backend()

        def backend(graph_module, example_inputs):
            return graph_module.forward

        try:
            compiler.set_default_backend(backend)
            expected_queries = (
                compiler.is_compiling(),
                compiler.is_dynamo_compiling(),
                compiler.is_exporting(),
            )
            original_guard_collectives = compiler.set_enable_guard_collectives(False)

            try:
                @compiler.allow_in_graph
                def state(value):
                    return (
                        value + 1,
                        compiler.is_compiling(),
                        compiler.is_dynamo_compiling(),
                        compiler.is_exporting(),
                    )

                self.assertEqual(state(1), (2, False, False, False))
                self.assertIs(compiler.get_default_backend(), backend)
                self.assertIs(compiler.set_enable_guard_collectives(False), False)
                self.assertEqual(
                    (
                        compiler.is_compiling(),
                        compiler.is_dynamo_compiling(),
                        compiler.is_exporting(),
                    ),
                    expected_queries,
                )
            finally:
                compiler.set_enable_guard_collectives(original_guard_collectives)
        finally:
            compiler.set_default_backend(original_backend)

        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch, "export"))
        self.assertFalse(hasattr(torch.compiler, "compile"))
        self.assertFalse(hasattr(torch.compiler, "register_backend"))
        self.assertFalse(hasattr(torch.compiler, "substitute_in_graph"))
        self.assertFalse(hasattr(torch.compiler, "cudagraph_mark_step_begin"))

    def test_import_and_marking_do_not_import_pytorch_or_dynamo(self):
        script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

calls = []

@torch.compiler.allow_in_graph
def function(value):
    calls.append(value)
    return value + 1

modules_before_call = set(sys.modules)
allowed_len = torch.compiler.allow_in_graph(len)
assert function(1) == 2
assert function(2) == 3
assert allowed_len([1, 2, 3]) == 3
assert calls == [1, 2]
assert not hasattr(function, "_dynamo_marked_constant")
assert not hasattr(function, "_torchdynamo_disable")
assert (
    torch.compiler.is_compiling(),
    torch.compiler.is_dynamo_compiling(),
    torch.compiler.is_exporting(),
) == (False, False, False)
assert set(sys.modules) == modules_before_call
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
assert not any(
    name.startswith("torch_rs._dynamo")
    or name.startswith("torch_rs.compiler.backends")
    or name == "torch_rs.compiler.registry"
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
