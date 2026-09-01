import copy
import gc
import importlib
import inspect
import pickle
import subprocess
import sys
import types
import typing
import unittest
import weakref

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
        self.calls = []

    def __call__(self, value):
        self.calls.append(value)
        return value + len(self.calls)


class _SlotCallable:
    __slots__ = ()

    def __call__(self):
        return "called"


class CompilerAllowInGraphTests(unittest.TestCase):
    def test_returns_function_lambda_builtin_and_callable_objects_without_wrapping(self):
        calls = []

        def calculate(value: int, *, scale: int = 1) -> int:
            """Calculate eagerly."""
            calls.append((value, scale))
            return value * scale + len(calls)

        shared_attribute = []
        calculate.custom_attribute = shared_attribute
        original_dict = dict(calculate.__dict__)

        marked = torch.compiler.allow_in_graph(calculate)

        self.assertIs(marked, calculate)
        self.assertIn(
            id(calculate),
            torch.compiler._state._allowed_in_graph_callable_ids,
        )
        self.assertEqual(calculate.__dict__, original_dict)
        self.assertEqual(marked(3, scale=2), 7)
        self.assertEqual(marked(3, scale=2), 8)
        self.assertEqual(calls, [(3, 2), (3, 2)])
        self.assertEqual(
            str(inspect.signature(marked)),
            "(value: int, *, scale: int = 1) -> int",
        )
        self.assertEqual(marked.__name__, "calculate")
        self.assertIn("<locals>.calculate", marked.__qualname__)
        self.assertEqual(marked.__module__, __name__)
        self.assertEqual(marked.__doc__, "Calculate eagerly.")
        self.assertIs(marked.custom_attribute, shared_attribute)

        lambda_function = lambda value: value * 2
        self.assertEqual(lambda_function.__dict__, {})
        marked_lambda = torch.compiler.allow_in_graph(lambda_function)
        self.assertIs(marked_lambda, lambda_function)
        self.assertIn(
            id(lambda_function),
            torch.compiler._state._allowed_in_graph_callable_ids,
        )
        self.assertEqual(lambda_function.__dict__, {})
        self.assertEqual(marked_lambda(4), 8)
        self.assertEqual(marked_lambda.__name__, "<lambda>")

        self.assertIs(torch.compiler.allow_in_graph(len), len)
        self.assertEqual(len([1, 2, 3]), 3)

        callable_object = _CallableObject()
        self.assertEqual(callable_object.__dict__, {"calls": []})
        marked_object = torch.compiler.allow_in_graph(callable_object)
        self.assertIs(marked_object, callable_object)
        self.assertIn(
            id(callable_object),
            torch.compiler._state._allowed_in_graph_callable_ids,
        )
        self.assertEqual(callable_object.__dict__, {"calls": []})
        self.assertEqual(marked_object(5), 6)
        self.assertEqual(callable_object.calls, [5])

    def test_marker_registry_is_weakref_backed_and_survives_compiler_reload(self):
        compiler = torch.compiler
        target = _CallableObject()
        target_id = id(target)
        target_ref = weakref.ref(target)

        self.assertNotIn(target_id, compiler._state._allowed_in_graph_callable_ids)
        self.assertIs(compiler.allow_in_graph(target), target)
        self.assertIn(target_id, compiler._state._allowed_in_graph_callable_ids)

        reloaded = importlib.reload(compiler)
        self.assertIs(reloaded, compiler)
        self.assertIn(target_id, compiler._state._allowed_in_graph_callable_ids)

        del target
        for _ in range(10):
            if target_ref() is None:
                break
            gc.collect()
        self.assertIsNone(target_ref())
        self.assertNotIn(target_id, compiler._state._allowed_in_graph_callable_ids)

    def test_sequence_inputs_return_fresh_lists_with_original_callables(self):
        def first():
            return "first"

        second = lambda: "second"
        for target in ([first, second, len], (first, second, len)):
            with self.subTest(type=type(target)):
                result = torch.compiler.allow_in_graph(target)
                self.assertIs(type(result), list)
                self.assertIsNot(result, target)
                self.assertEqual(len(result), len(target))
                for actual, expected in zip(result, target):
                    self.assertIs(actual, expected)

        self.assertEqual(torch.compiler.allow_in_graph([]), [])
        self.assertEqual(torch.compiler.allow_in_graph(()), [])

    def test_rejects_non_callable_targets(self):
        cases = (None, 1, object(), types.SimpleNamespace(), [1], (1,))
        for target in cases:
            with self.subTest(target=target):
                with self.assertRaises(AssertionError) as raised:
                    torch.compiler.allow_in_graph(target)
                self.assertEqual(str(raised.exception), "allow_in_graph expects a callable")
                self.assertEqual(raised.exception.args, ("allow_in_graph expects a callable",))

    def test_rejects_non_weakrefable_callable_targets_like_pytorch_2_13(self):
        cases = (
            (
                _SlotCallable(),
                "cannot create weak reference to '_SlotCallable' object",
            ),
            (
                list.append,
                "cannot create weak reference to 'method_descriptor' object",
            ),
        )
        for target, message in cases:
            with self.subTest(target=target):
                with self.assertRaises(TypeError) as raised:
                    torch.compiler.allow_in_graph(target)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_signature_documentation_and_module_identity(self):
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
        self.assertNotIn("torch", function.__code__.co_names)
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

    def test_exports_wildcards_copy_pickle_and_reload_use_canonical_objects(self):
        compiler = torch.compiler
        marker = compiler.allow_in_graph

        self.assertEqual(compiler.__all__, COMPILER_EXPORTS)
        from torch_rs.compiler import allow_in_graph

        self.assertIs(allow_in_graph, marker)

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

        for function in (marker, _picklable_allowed_function, len):
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(function=function.__name__, protocol=protocol):
                    self.assertIs(
                        pickle.loads(pickle.dumps(function, protocol=protocol)),
                        function,
                    )

        self.assertEqual(_picklable_allowed_function(4, increment=3), 7)
        self.assertEqual(_picklable_allowed_function.__dict__, {})

        old_marker = marker
        old_exports = compiler.__all__
        reloaded = importlib.reload(compiler)
        new_marker = reloaded.allow_in_graph

        self.assertIs(reloaded, compiler)
        self.assertIs(torch.compiler, compiler)
        self.assertIs(sys.modules["torch_rs.compiler"], compiler)
        self.assertIsNot(new_marker, old_marker)
        self.assertIsNot(compiler.__all__, old_exports)
        self.assertEqual(compiler.__all__, COMPILER_EXPORTS)
        self.assertIs(new_marker(len), len)
        self.assertIs(copy.copy(old_marker), old_marker)
        self.assertIs(copy.deepcopy(old_marker), old_marker)
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(old_marker)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            self.assertIs(
                pickle.loads(pickle.dumps(new_marker, protocol=protocol)),
                new_marker,
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

    def test_marker_does_not_enable_compilation_backend_registration_or_cuda(self):
        compiler = torch.compiler
        original_backend = compiler.get_default_backend()
        original_collectives = compiler.set_enable_guard_collectives(False)

        def backend(graph_module, example_inputs):
            return graph_module.forward

        try:
            compiler.set_default_backend(backend)

            @compiler.allow_in_graph
            def function(value):
                return (
                    value + 1,
                    compiler.is_compiling(),
                    compiler.is_dynamo_compiling(),
                    compiler.is_exporting(),
                    torch.is_grad_enabled(),
                )

            self.assertEqual(function(4), (5, False, False, False, True))
            with torch.no_grad():
                self.assertEqual(function(4), (5, False, False, False, False))
            self.assertIs(compiler.get_default_backend(), backend)
            self.assertEqual(compiler.list_backends(), [])
            self.assertFalse(hasattr(torch, "compile"))
            self.assertFalse(hasattr(torch, "export"))
            self.assertFalse(hasattr(torch, "_dynamo"))
            self.assertFalse(hasattr(compiler, "compile"))
            self.assertFalse(hasattr(compiler, "substitute_in_graph"))
            self.assertFalse(hasattr(compiler, "register_backend"))
            self.assertIs(torch.cuda.is_available(), False)
            self.assertIs(torch.backends.cuda.is_built(), False)
        finally:
            compiler.set_default_backend(original_backend)
            compiler.set_enable_guard_collectives(original_collectives)

    def test_import_and_marking_do_not_import_pytorch(self):
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
    return value + 1

original_dict = dict(function.__dict__)
assert torch.compiler.allow_in_graph(function) is function
assert function.__dict__ == original_dict
assert function(1) == 2
assert torch.compiler.allow_in_graph(len) is len
try:
    torch.compiler.allow_in_graph(None)
except AssertionError as error:
    assert error.args == ("allow_in_graph expects a callable",)
else:
    raise AssertionError("allow_in_graph accepted None")
assert not hasattr(torch, "compile")
assert not hasattr(torch.compiler, "substitute_in_graph")
assert torch.compiler.list_backends() == []
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
