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


@torch.compiler.allow_in_graph
def _picklable_allowed_function(value, *, increment=1):
    return value + increment


class _TrackingList(list):
    def __init__(self, values):
        super().__init__(values)
        self.visited = []

    def __iter__(self):
        for index, value in enumerate(super().__iter__()):
            self.visited.append(index)
            yield value


class CompilerAllowInGraphTests(unittest.TestCase):
    def test_decorator_preserves_identity_metadata_and_eager_calls(self):
        calls = []

        def calculate(value: int, *, scale: int = 1) -> int:
            """Calculate eagerly."""
            calls.append((value, scale))
            return value * scale + len(calls)

        calculate.custom_metadata = object()
        original = calculate
        original_dict = calculate.__dict__.copy()
        decorated = torch.compiler.allow_in_graph(calculate)

        self.assertIs(decorated, original)
        self.assertEqual(decorated.__dict__, original_dict)
        self.assertEqual(decorated(3, scale=2), 7)
        self.assertEqual(decorated(3, scale=2), 8)
        self.assertEqual(calls, [(3, 2), (3, 2)])
        self.assertEqual(
            str(inspect.signature(decorated)),
            "(value: int, *, scale: int = 1) -> int",
        )
        self.assertEqual(decorated.__name__, "calculate")
        self.assertEqual(decorated.__doc__, "Calculate eagerly.")

    def test_decorated_and_direct_bound_methods_preserve_binding(self):
        class Accumulator:
            def __init__(self):
                self.total = 0

            @torch.compiler.allow_in_graph
            def add(self, value):
                self.total += value
                return self.total

            @classmethod
            @torch.compiler.allow_in_graph
            def identify(cls, value):
                return cls, value

            @staticmethod
            @torch.compiler.allow_in_graph
            def double(value):
                return value * 2

        left = Accumulator()
        right = Accumulator()

        self.assertIsInstance(left.add, types.MethodType)
        self.assertIs(left.add.__self__, left)
        self.assertIs(left.add.__func__, Accumulator.add)
        self.assertEqual(left.add(2), 2)
        self.assertEqual(left.add(3), 5)
        self.assertEqual(right.add(7), 7)
        self.assertEqual(Accumulator.identify("value"), (Accumulator, "value"))
        self.assertEqual(Accumulator.double(4), 8)

        bound = left.add
        self.assertIs(torch.compiler.allow_in_graph(bound), bound)
        self.assertEqual(bound(1), 6)

    def test_lists_tuples_and_nested_sequences_become_fresh_lists(self):
        def first(value):
            return value + 1

        def second(value):
            return value * 2

        source_list = [first, second]
        from_list = torch.compiler.allow_in_graph(source_list)
        self.assertIs(type(from_list), list)
        self.assertIsNot(from_list, source_list)
        self.assertIs(from_list[0], first)
        self.assertIs(from_list[1], second)

        source_tuple = (first, second)
        from_tuple = torch.compiler.allow_in_graph(source_tuple)
        self.assertIs(type(from_tuple), list)
        self.assertIs(from_tuple[0], first)
        self.assertIs(from_tuple[1], second)

        nested_source = (first, [second, (first,)])
        nested = torch.compiler.allow_in_graph(nested_source)
        self.assertIs(type(nested), list)
        self.assertIs(nested[0], first)
        self.assertIs(type(nested[1]), list)
        self.assertIsNot(nested[1], nested_source[1])
        self.assertIs(nested[1][0], second)
        self.assertIs(type(nested[1][1]), list)
        self.assertIs(nested[1][1][0], first)

        empty_list = []
        empty_tuple = ()
        self.assertEqual(torch.compiler.allow_in_graph(empty_list), [])
        self.assertIsNot(torch.compiler.allow_in_graph(empty_list), empty_list)
        self.assertEqual(torch.compiler.allow_in_graph(empty_tuple), [])

    def test_noncallables_fail_left_to_right_with_pytorch_error(self):
        message = "allow_in_graph expects a callable"
        for target in (None, 1, "callable", object(), iter(())):
            with self.subTest(target=target):
                with self.assertRaises(AssertionError) as raised:
                    torch.compiler.allow_in_graph(target)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

        def first():
            return "first"

        def unvisited():
            return "unvisited"

        entries = _TrackingList([first, None, unvisited])
        with self.assertRaises(AssertionError) as raised:
            torch.compiler.allow_in_graph(entries)
        self.assertEqual(str(raised.exception), message)
        self.assertEqual(entries.visited, [0, 1])

        nested_entries = _TrackingList([first, 3, unvisited])
        with self.assertRaises(AssertionError) as raised:
            torch.compiler.allow_in_graph((first, nested_entries, unvisited))
        self.assertEqual(str(raised.exception), message)
        self.assertEqual(nested_entries.visited, [0, 1])

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
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_exports_copying_and_pickling_use_canonical_objects(self):
        compiler = torch.compiler
        function = compiler.allow_in_graph

        self.assertEqual(
            compiler.__all__,
            [
                "assume_constant_result",
                "reset",
                "allow_in_graph",
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
                "skip_guard_on_globals_unsafe",
                "skip_all_guards_unsafe",
            ],
        )
        compiler_namespace = {}
        exec("from torch_rs.compiler import *", compiler_namespace)
        self.assertEqual(
            {name for name in compiler_namespace if not name.startswith("__")},
            set(compiler.__all__),
        )
        for name in compiler.__all__:
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
        self.assertEqual(_picklable_allowed_function.__dict__, {})

    def test_call_shape_errors_match_pytorch_2_13(self):
        allow_in_graph = torch.compiler.allow_in_graph
        function = lambda: None
        cases = (
            (
                lambda: allow_in_graph(),
                "allow_in_graph() missing 1 required positional argument: 'fn'",
            ),
            (
                lambda: allow_in_graph(function, function),
                "allow_in_graph() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: allow_in_graph(function, fn=function),
                "allow_in_graph() got multiple values for argument 'fn'",
            ),
            (
                lambda: allow_in_graph(function, extra=True),
                "allow_in_graph() got an unexpected keyword argument 'extra'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_reload_replaces_only_the_module_function(self):
        compiler = torch.compiler
        old_function = compiler.allow_in_graph
        old_exports = compiler.__all__
        reloaded = importlib.reload(compiler)
        new_function = reloaded.allow_in_graph

        self.assertIs(reloaded, compiler)
        self.assertIs(torch.compiler, compiler)
        self.assertIs(sys.modules["torch_rs.compiler"], compiler)
        self.assertIsNot(new_function, old_function)
        self.assertIsNot(compiler.__all__, old_exports)
        self.assertIn("allow_in_graph", compiler.__all__)
        self.assertEqual(_picklable_allowed_function(2, increment=3), 5)
        self.assertIs(
            new_function(_picklable_allowed_function),
            _picklable_allowed_function,
        )
        self.assertIs(copy.copy(old_function), old_function)
        self.assertIs(copy.deepcopy(old_function), old_function)
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(old_function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            self.assertIs(
                pickle.loads(pickle.dumps(new_function, protocol)), new_function
            )

    def test_eager_boundary_does_not_initialize_a_compiler(self):
        def function():
            return (
                torch.compiler.is_compiling(),
                torch.compiler.is_dynamo_compiling(),
                torch.compiler.is_exporting(),
            )

        before = function.__dict__.copy()
        self.assertIs(torch.compiler.allow_in_graph(function), function)
        self.assertEqual(function.__dict__, before)
        self.assertEqual(function(), (False, False, False))
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch, "export"))
        self.assertFalse(hasattr(torch.compiler, "compile"))

        script = r"""
import sys

class RejectCompilerImport:
    def find_spec(self, fullname, path=None, target=None):
        if (
            fullname == "torch"
            or fullname.startswith("torch.")
            or fullname == "torch_rs._dynamo"
            or fullname.startswith("torch_rs._dynamo.")
            or fullname.startswith("torch_rs.compiler.backends")
            or fullname == "torch_rs.compiler.registry"
        ):
            raise RuntimeError(f"compiler import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectCompilerImport())
import torch_rs as torch

modules_before_call = set(sys.modules)
calls = []

@torch.compiler.allow_in_graph
def function(value):
    calls.append(value)
    return value + 1

result = torch.compiler.allow_in_graph((function, [function]))
assert result == [function, [function]]
assert result[0] is function
assert result[1][0] is function
assert function.__dict__ == {}
assert function(1) == 2
assert function(2) == 3
assert calls == [1, 2]
assert torch.compiler.is_compiling() is False
assert torch.compiler.is_dynamo_compiling() is False
assert torch.compiler.is_exporting() is False
assert set(sys.modules) == modules_before_call
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
assert not any(
    name == "torch_rs._dynamo"
    or name.startswith("torch_rs._dynamo.")
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
