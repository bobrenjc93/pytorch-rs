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


def _actual_picklable_function(value, *, increment=1):
    return value + increment


_actual_picklable_function = torch.compiler.allow_in_graph(_actual_picklable_function)


def _reference_picklable_function(value, *, increment=1):
    return value + increment


if reference_torch is not None:
    _reference_picklable_function = reference_torch.compiler.allow_in_graph(
        _reference_picklable_function
    )


class _CallableObject:
    def __init__(self):
        self.calls = []

    def __call__(self, value):
        self.calls.append(value)
        return value + 3


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

    def callable_outcome(self, module):
        calls = []

        def function(value, *, scale=1):
            calls.append((value, scale))
            return value * scale + len(calls)

        function.custom_attribute = ["preserved"]
        original_dict = dict(function.__dict__)
        positional = module.compiler.allow_in_graph(function)
        keyword = module.compiler.allow_in_graph(fn=function)

        lambda_function = lambda value: value + 2
        lambda_result = module.compiler.allow_in_graph(lambda_function)

        callable_object = _CallableObject()
        object_dict = dict(callable_object.__dict__)
        object_result = module.compiler.allow_in_graph(callable_object)

        return (
            positional is function,
            keyword is function,
            function.__dict__ == original_dict,
            function(3, scale=2),
            function(3, scale=2),
            calls,
            str(inspect.signature(function)),
            function.__name__,
            "<locals>.function" in function.__qualname__,
            lambda_result is lambda_function,
            lambda_function(4),
            module.compiler.allow_in_graph(len) is len,
            module.compiler.allow_in_graph(len)([1, 2, 3]),
            object_result is callable_object,
            callable_object.__dict__ == object_dict,
            callable_object(5),
            callable_object.calls,
        )

    def test_callable_identity_and_eager_invocation_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_outcome(torch),
            self.callable_outcome(reference_torch),
        )

    def sequence_outcome(self, module):
        def function(value):
            return value + 1

        list_input = [function, len]
        from_list = module.compiler.allow_in_graph(list_input)
        from_tuple = module.compiler.allow_in_graph((function, len))
        return (
            type(from_list) is list,
            from_list is not list_input,
            from_list[0] is function,
            from_list[1] is len,
            type(from_tuple) is list,
            from_tuple[0] is function,
            from_tuple[1] is len,
            from_list[0](1),
            from_tuple[1]([1, 2]),
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
            lambda: "not callable",
            lambda: types.SimpleNamespace(existing="preserved"),
            lambda: [lambda: None, 1],
        )
        for case, target_factory in enumerate(target_factories):
            with self.subTest(case=case):
                actual_target = target_factory()
                expected_target = target_factory()
                actual_before = dict(getattr(actual_target, "__dict__", {}))
                expected_before = dict(getattr(expected_target, "__dict__", {}))
                self.assert_error_matches(
                    lambda: torch.compiler.allow_in_graph(actual_target),
                    lambda: reference_torch.compiler.allow_in_graph(expected_target),
                )
                self.assertEqual(getattr(actual_target, "__dict__", {}), actual_before)
                self.assertEqual(
                    getattr(expected_target, "__dict__", {}),
                    expected_before,
                )

    def test_signature_and_function_metadata_match_pytorch_2_13(self):
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
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )
        self.assertIn("Dynamo", actual.__doc__)
        self.assertIn("torch.compile", actual.__doc__)
        self.assertIn("unsupported", actual.__doc__)

    def test_exports_wildcard_copy_and_pickle_match_pytorch_2_13(self):
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
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(pickle.loads(pickle.dumps(expected, protocol)), expected)
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

        for function in (
            _actual_picklable_function,
            _reference_picklable_function,
        ):
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            self.assertEqual(function(4, increment=3), 7)
            self.assertEqual(function.__dict__, {})
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(marked=function.__name__, protocol=protocol):
                    self.assertIs(
                        pickle.loads(pickle.dumps(function, protocol)),
                        function,
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

    def reload_outcome(self, package_name, compiler_module_name):
        script = f"""
import copy
import importlib
import json
import pickle

import {package_name} as module

compiler = importlib.import_module({compiler_module_name!r})
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

    print(json.dumps([
        reloaded is compiler,
        module.compiler is compiler,
        old_function is new_function,
        old_exports is compiler.__all__,
        compiler.get_default_backend() is backend,
        new_function(len) is len,
        copy.copy(old_function) is old_function,
        copy.deepcopy(old_function) is old_function,
        old_pickle_error,
        [
            pickle.loads(pickle.dumps(new_function, protocol)) is new_function
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
        ],
    ]))
finally:
    compiler.set_default_backend(original_backend)
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
        self.assertEqual(
            self.reload_outcome("torch_rs", "torch_rs.compiler"),
            self.reload_outcome("torch", "torch.compiler"),
        )

    def test_graph_execution_and_related_compiler_apis_remain_unsupported(self):
        @torch.compiler.allow_in_graph
        def state():
            return (
                torch.compiler.is_compiling(),
                torch.compiler.is_dynamo_compiling(),
                torch.compiler.is_exporting(),
            )

        self.assertEqual(state(), (False, False, False))
        self.assertTrue(callable(reference_torch.compile))
        self.assertTrue(callable(reference_torch.compiler.compile))
        self.assertTrue(callable(reference_torch.compiler.substitute_in_graph))
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch, "export"))
        self.assertFalse(hasattr(torch.compiler, "compile"))
        self.assertFalse(hasattr(torch.compiler, "register_backend"))
        self.assertFalse(hasattr(torch.compiler, "substitute_in_graph"))
        self.assertFalse(hasattr(torch.compiler, "cudagraph_mark_step_begin"))


if __name__ == "__main__":
    unittest.main()
