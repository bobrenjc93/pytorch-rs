import copy
import importlib
import inspect
import json
import pickle
import pickletools
import subprocess
import sys
import types
import typing
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@torch.compiler.allow_in_graph
def _actual_picklable_function(value):
    return value + 1


def _reference_picklable_function(value):
    return value + 1


if reference_torch is not None:
    _reference_picklable_function = reference_torch.compiler.allow_in_graph(
        _reference_picklable_function
    )


class _CallableObject:
    def __init__(self):
        self.calls = []

    def __call__(self, value):
        self.calls.append(value)
        return value + len(self.calls)


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CompilerAllowInGraphReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "compiler.allow_in_graph differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def pickle_shape(self, function, protocol):
        shape = []
        for opcode, argument, _ in pickletools.genops(
            pickle.dumps(function, protocol=protocol)
        ):
            if opcode.name == "FRAME":
                argument = "<frame length>"
            elif isinstance(argument, str):
                argument = argument.replace("torch_rs", "torch")
            shape.append((opcode.name, argument))
        return shape

    def callable_outcome(self, module):
        calls = []

        def calculate(value: int, *, scale: int = 1) -> int:
            """Calculate eagerly."""
            calls.append((value, scale))
            return value * scale + len(calls)

        shared_attribute = []
        calculate.custom_attribute = shared_attribute
        original_dict = dict(calculate.__dict__)
        marked = module.compiler.allow_in_graph(calculate)

        lambda_function = lambda value: value * 2
        lambda_dict = dict(lambda_function.__dict__)
        marked_lambda = module.compiler.allow_in_graph(lambda_function)

        callable_object = _CallableObject()
        object_dict = dict(callable_object.__dict__)
        marked_object = module.compiler.allow_in_graph(callable_object)

        return (
            marked is calculate,
            calculate.__dict__ == original_dict,
            marked(3, scale=2),
            marked(3, scale=2),
            calls,
            str(inspect.signature(marked)),
            marked.__name__,
            "<locals>.calculate" in marked.__qualname__,
            marked.__module__,
            marked.__doc__,
            marked.__annotations__,
            marked.__defaults__,
            marked.__kwdefaults__,
            marked.custom_attribute is shared_attribute,
            marked_lambda is lambda_function,
            lambda_function.__dict__ == lambda_dict,
            marked_lambda(4),
            marked_lambda.__name__,
            module.compiler.allow_in_graph(len) is len,
            len([1, 2, 3]),
            marked_object is callable_object,
            callable_object.__dict__ == object_dict,
            marked_object(5),
            callable_object.calls,
        )

    def test_function_lambda_builtin_and_callable_object_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_outcome(torch),
            self.callable_outcome(reference_torch),
        )

    def sequence_outcome(self, module):
        def first():
            return "first"

        second = lambda: "second"
        list_input = [first, second, len]
        tuple_input = (first, second, len)
        list_result = module.compiler.allow_in_graph(list_input)
        tuple_result = module.compiler.allow_in_graph(tuple_input)
        empty_list_result = module.compiler.allow_in_graph([])
        empty_tuple_result = module.compiler.allow_in_graph(())
        return (
            type(list_result) is list,
            list_result is not list_input,
            [actual is expected for actual, expected in zip(list_result, list_input)],
            type(tuple_result) is list,
            tuple_result is not tuple_input,
            [actual is expected for actual, expected in zip(tuple_result, tuple_input)],
            empty_list_result,
            empty_tuple_result,
        )

    def test_sequence_inputs_match_pytorch_2_13(self):
        self.assertEqual(
            self.sequence_outcome(torch),
            self.sequence_outcome(reference_torch),
        )

    def test_non_callable_errors_match_pytorch_2_13(self):
        target_factories = (
            lambda: None,
            lambda: 1,
            object,
            types.SimpleNamespace,
            lambda: [1],
            lambda: (1,),
        )
        for case, target_factory in enumerate(target_factories):
            with self.subTest(case=case):
                actual_target = target_factory()
                expected_target = target_factory()
                self.assert_error_matches(
                    lambda: torch.compiler.allow_in_graph(actual_target),
                    lambda: reference_torch.compiler.allow_in_graph(expected_target),
                )

    def test_signature_documentation_and_ownership_match_pytorch_2_13(self):
        actual_compiler = importlib.import_module("torch_rs.compiler")
        expected_compiler = importlib.import_module("torch.compiler")
        actual = actual_compiler.allow_in_graph
        expected = expected_compiler.allow_in_graph

        self.assertIs(torch.compiler, actual_compiler)
        self.assertIs(reference_torch.compiler, expected_compiler)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual)), str(inspect.signature(expected))
        )
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(typing.get_type_hints(actual), typing.get_type_hints(expected))
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertIs(inspect.getmodule(actual), actual_compiler)
        self.assertIs(inspect.getmodule(expected), expected_compiler)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )

    def test_exports_wildcards_copy_pickle_and_reload_match_pytorch_2_13(self):
        actual_compiler = torch.compiler
        expected_compiler = reference_torch.compiler
        actual = actual_compiler.allow_in_graph
        expected = expected_compiler.allow_in_graph
        supported = {
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
        }

        self.assertEqual(
            actual_compiler.__all__,
            [name for name in expected_compiler.__all__ if name in supported],
        )
        self.assertEqual(
            torch.__all__.count("compiler"),
            reference_torch.__all__.count("compiler"),
        )
        self.assertEqual(
            torch.__all__.count("allow_in_graph"),
            reference_torch.__all__.count("allow_in_graph"),
        )

        for module in (actual_compiler, expected_compiler):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            for name in supported:
                self.assertIs(namespace[name], getattr(module, name))

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("compiler", namespace)
            self.assertNotIn("allow_in_graph", namespace)

        for function in (actual, expected):
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)

        for function in (_actual_picklable_function, _reference_picklable_function):
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            self.assertEqual(function(4), 5)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(pickle.loads(pickle.dumps(expected, protocol)), expected)
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )
                self.assertIs(
                    pickle.loads(pickle.dumps(_actual_picklable_function, protocol)),
                    _actual_picklable_function,
                )
                self.assertIs(
                    pickle.loads(pickle.dumps(_reference_picklable_function, protocol)),
                    _reference_picklable_function,
                )

        old_function = actual_compiler.allow_in_graph
        old_exports = actual_compiler.__all__
        reloaded = importlib.reload(actual_compiler)
        actual_outcome = (
            reloaded is actual_compiler,
            torch.compiler is actual_compiler,
            old_function is actual_compiler.allow_in_graph,
            old_exports is actual_compiler.__all__,
            actual_compiler.allow_in_graph(len) is len,
            actual_compiler.__all__,
        )

        script = r"""
import importlib
import json
import torch

compiler = torch.compiler
old_function = compiler.allow_in_graph
old_exports = compiler.__all__
reloaded = importlib.reload(compiler)
print(json.dumps({
    "same_module": reloaded is compiler,
    "module_binding": torch.compiler is compiler,
    "same_function": old_function is compiler.allow_in_graph,
    "same_exports": old_exports is compiler.__all__,
    "marker_builtin": compiler.allow_in_graph(len) is len,
    "all": compiler.__all__,
}))
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
        expected_reload = json.loads(completed.stdout)
        expected_outcome = (
            expected_reload["same_module"],
            expected_reload["module_binding"],
            expected_reload["same_function"],
            expected_reload["same_exports"],
            expected_reload["marker_builtin"],
            expected_reload["all"],
        )

        self.assertEqual(actual_outcome[:-1], expected_outcome[:-1])
        self.assertEqual(
            actual_outcome[-1],
            [name for name in expected_outcome[-1] if name in supported],
        )

    def test_call_shape_errors_match_pytorch_2_13(self):
        actual = torch.compiler.allow_in_graph
        expected = reference_torch.compiler.allow_in_graph
        actual_function = lambda: None
        expected_function = lambda: None
        cases = (
            (lambda: actual(), lambda: expected()),
            (
                lambda: actual(actual_function, actual_function),
                lambda: expected(expected_function, expected_function),
            ),
            (
                lambda: actual(actual_function, fn=actual_function),
                lambda: expected(expected_function, fn=expected_function),
            ),
            (
                lambda: actual(actual_function, extra=True),
                lambda: expected(expected_function, extra=True),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_graph_dynamo_backend_registration_and_cuda_compile_stay_unsupported(self):
        @torch.compiler.allow_in_graph
        def function(value):
            return (
                value + 1,
                torch.compiler.is_compiling(),
                torch.compiler.is_dynamo_compiling(),
                torch.compiler.is_exporting(),
            )

        self.assertEqual(function(4), (5, False, False, False))
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch, "export"))
        self.assertFalse(hasattr(torch, "_dynamo"))
        self.assertFalse(hasattr(torch.compiler, "compile"))
        self.assertFalse(hasattr(torch.compiler, "substitute_in_graph"))
        self.assertFalse(hasattr(torch.compiler, "register_backend"))
        self.assertEqual(torch.compiler.list_backends(), [])
        self.assertIs(torch.cuda.is_available(), False)
        self.assertIs(torch.backends.cuda.is_built(), False)

        self.assertTrue(callable(reference_torch.compile))
        self.assertTrue(callable(reference_torch.compiler.compile))
        self.assertTrue(callable(reference_torch.compiler.substitute_in_graph))
        self.assertTrue(callable(reference_torch._dynamo.register_backend))


if __name__ == "__main__":
    unittest.main()
