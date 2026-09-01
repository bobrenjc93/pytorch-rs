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
def _actual_picklable_function(value, *, increment=1):
    return value + increment


def _reference_picklable_function(value, *, increment=1):
    return value + increment


if reference_torch is not None:
    _reference_picklable_function = reference_torch.compiler.allow_in_graph(
        _reference_picklable_function
    )


SUPPORTED_COMPILER_EXPORTS = {
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


class _CallableObject:
    def __init__(self):
        self.existing = "preserved"

    def __call__(self, value):
        return value + 2


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CompilerAllowInGraphReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "compiler.allow_in_graph differentials require pinned "
                "PyTorch 2.13.0"
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

    def identity_and_call_outcome(self, module):
        calls = []

        @module.compiler.allow_in_graph
        def calculate(value, *, scale=1):
            calls.append((value, scale))
            return value * scale + len(calls)

        anonymous = lambda value: value + 2
        callable_object = _CallableObject()

        function_dict = dict(calculate.__dict__)
        callable_dict = dict(callable_object.__dict__)

        direct = module.compiler.allow_in_graph(calculate)
        keyword = module.compiler.allow_in_graph(fn=calculate)
        allowed_lambda = module.compiler.allow_in_graph(anonymous)
        allowed_builtin = module.compiler.allow_in_graph(len)
        allowed_object = module.compiler.allow_in_graph(callable_object)
        input_list = [calculate, anonymous, len]
        input_tuple = (calculate, anonymous)
        from_list = module.compiler.allow_in_graph(input_list)
        from_tuple = module.compiler.allow_in_graph(input_tuple)

        return (
            direct is calculate,
            keyword is calculate,
            allowed_lambda is anonymous,
            allowed_builtin is len,
            allowed_object is callable_object,
            from_list == [calculate, anonymous, len],
            from_list is not input_list,
            from_tuple == [calculate, anonymous],
            type(from_tuple).__name__,
            calculate(3, scale=2),
            calculate(3, scale=2),
            calls,
            allowed_lambda(4),
            allowed_builtin([1, 2, 3]),
            allowed_object(4),
            str(inspect.signature(calculate)),
            calculate.__name__,
            calculate.__module__,
            calculate.__dict__ == function_dict,
            callable_object.__dict__ == callable_dict,
            hasattr(calculate, "__wrapped__"),
            hasattr(calculate, "_dynamo_marked_constant"),
            hasattr(calculate, "_torchdynamo_disable"),
        )

    def test_identity_and_eager_behavior_match_pytorch_2_13(self):
        self.assertEqual(
            self.identity_and_call_outcome(torch),
            self.identity_and_call_outcome(reference_torch),
        )

    def test_noncallable_errors_match_pytorch_2_13(self):
        target_factories = (
            lambda: None,
            lambda: 1,
            object,
            types.SimpleNamespace,
            lambda: "text",
            lambda: [1],
        )
        for case, target_factory in enumerate(target_factories):
            with self.subTest(case=case):
                actual_target = target_factory()
                expected_target = target_factory()
                self.assert_error_matches(
                    lambda: torch.compiler.allow_in_graph(actual_target),
                    lambda: reference_torch.compiler.allow_in_graph(expected_target),
                )

    def test_signature_documentation_and_metadata_match_pytorch_2_13(self):
        actual_compiler = importlib.import_module("torch_rs.compiler")
        expected_compiler = importlib.import_module("torch.compiler")
        actual = actual_compiler.allow_in_graph
        expected = expected_compiler.allow_in_graph

        self.assertIs(torch.compiler, actual_compiler)
        self.assertIs(reference_torch.compiler, expected_compiler)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual)),
            str(inspect.signature(expected)),
        )
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(typing.get_type_hints(actual), typing.get_type_hints(expected))
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"),
            expected.__module__,
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

    def test_exports_wildcard_copy_pickle_and_reload_match_pytorch_2_13(self):
        actual_compiler = torch.compiler
        expected_compiler = reference_torch.compiler
        actual = actual_compiler.allow_in_graph
        expected = expected_compiler.allow_in_graph

        self.assertEqual(
            actual_compiler.__all__,
            [
                name
                for name in expected_compiler.__all__
                if name in SUPPORTED_COMPILER_EXPORTS
            ],
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
            exec(f"from {module.__name__} import allow_in_graph", namespace)
            self.assertIs(namespace["allow_in_graph"], module.allow_in_graph)

            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            for name in SUPPORTED_COMPILER_EXPORTS:
                self.assertIs(namespace[name], getattr(module, name))

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("compiler", namespace)
            self.assertNotIn("allow_in_graph", namespace)

        for function in (actual, expected):
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(pickle.loads(pickle.dumps(expected, protocol)), expected)
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

        for function in (_actual_picklable_function, _reference_picklable_function):
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            self.assertEqual(function(4, increment=3), 7)
            self.assertFalse(hasattr(function, "__wrapped__"))
            self.assertFalse(hasattr(function, "_dynamo_marked_constant"))
            self.assertFalse(hasattr(function, "_torchdynamo_disable"))
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(marked=function.__name__, protocol=protocol):
                    self.assertIs(
                        pickle.loads(pickle.dumps(function, protocol)),
                        function,
                    )

    def reload_outcome(self, module, compiler_module_name):
        compiler = importlib.import_module(compiler_module_name)
        original_backend = compiler.get_default_backend()

        def backend(graph_module, example_inputs):
            return graph_module.forward

        try:
            compiler.set_default_backend(backend)
            old_function = compiler.allow_in_graph
            old_exports = compiler.__all__
            reloaded = importlib.reload(compiler)
            new_function = reloaded.allow_in_graph

            try:
                pickle.dumps(old_function)
            except BaseException as error:
                old_pickle_error = (
                    type(error).__name__,
                    "not the same object" in str(error),
                )
            else:
                old_pickle_error = None

            new_pickle_results = tuple(
                pickle.loads(pickle.dumps(new_function, protocol)) is new_function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            )

            return {
                "reloaded_is_compiler": reloaded is compiler,
                "module_compiler_is_compiler": module.compiler is compiler,
                "old_function_is_new_function": old_function is new_function,
                "old_exports_is_current_exports": old_exports is compiler.__all__,
                "backend_preserved": compiler.get_default_backend() is backend,
                "new_function_allows_builtin": new_function(len) is len,
                "old_copy_identity": copy.copy(old_function) is old_function,
                "old_deepcopy_identity": copy.deepcopy(old_function) is old_function,
                "old_pickle_error": old_pickle_error,
                "new_pickle_results": new_pickle_results,
            }
        finally:
            compiler.set_default_backend(original_backend)

    def reference_reload_outcome(self):
        script = r"""
import copy
import importlib
import json
import pickle
import torch

compiler = importlib.import_module("torch.compiler")
original_backend = compiler.get_default_backend()

def backend(graph_module, example_inputs):
    return graph_module.forward

try:
    compiler.set_default_backend(backend)
    old_function = compiler.allow_in_graph
    old_exports = compiler.__all__
    reloaded = importlib.reload(compiler)
    new_function = reloaded.allow_in_graph

    try:
        pickle.dumps(old_function)
    except BaseException as error:
        old_pickle_error = [
            type(error).__name__,
            "not the same object" in str(error),
        ]
    else:
        old_pickle_error = None

    new_pickle_results = [
        pickle.loads(pickle.dumps(new_function, protocol)) is new_function
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
    ]

    outcome = {
        "reloaded_is_compiler": reloaded is compiler,
        "module_compiler_is_compiler": torch.compiler is compiler,
        "old_function_is_new_function": old_function is new_function,
        "old_exports_is_current_exports": old_exports is compiler.__all__,
        "backend_preserved": compiler.get_default_backend() is backend,
        "new_function_allows_builtin": new_function(len) is len,
        "old_copy_identity": copy.copy(old_function) is old_function,
        "old_deepcopy_identity": copy.deepcopy(old_function) is old_function,
        "old_pickle_error": old_pickle_error,
        "new_pickle_results": new_pickle_results,
    }
finally:
    compiler.set_default_backend(original_backend)

print(json.dumps(outcome, sort_keys=True))
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
        return json.loads(completed.stdout)

    def test_reload_behavior_matches_pytorch_2_13(self):
        actual = self.reload_outcome(torch, "torch_rs.compiler")
        actual["old_pickle_error"] = list(actual["old_pickle_error"])
        actual["new_pickle_results"] = list(actual["new_pickle_results"])
        self.assertEqual(actual, self.reference_reload_outcome())

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

    def test_compiler_execution_boundaries_remain_unsupported(self):
        self.assertTrue(callable(reference_torch.compile))
        self.assertTrue(callable(reference_torch.compiler.compile))
        self.assertTrue(callable(reference_torch.compiler.substitute_in_graph))
        self.assertTrue(callable(reference_torch.compiler.cudagraph_mark_step_begin))
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch, "export"))
        self.assertFalse(hasattr(torch.compiler, "compile"))
        self.assertFalse(hasattr(torch.compiler, "register_backend"))
        self.assertFalse(hasattr(torch.compiler, "substitute_in_graph"))
        self.assertFalse(hasattr(torch.compiler, "cudagraph_mark_step_begin"))


if __name__ == "__main__":
    unittest.main()
