import contextlib
import copy
import importlib
import inspect
import pickle
import pickletools
import types
import typing
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


SUPPORTED_COMPILER_EXPORTS = {
    "assume_constant_result",
    "reset",
    "disable",
    "set_default_backend",
    "get_default_backend",
    "set_enable_guard_collectives",
    "cudagraph_mark_step_begin",
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


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CompilerCudagraphMarkStepBeginReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "compiler.cudagraph_mark_step_begin differentials require "
                "pinned PyTorch 2.13.0"
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

    def test_signature_annotations_documentation_and_identity_match(self):
        actual_compiler = importlib.import_module("torch_rs.compiler")
        expected_compiler = importlib.import_module("torch.compiler")
        actual = actual_compiler.cudagraph_mark_step_begin
        expected = expected_compiler.cudagraph_mark_step_begin

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

    def test_supported_state_and_return_value_match_pytorch_2_13(self):
        self.assertEqual(self.state_outcome(torch), self.state_outcome(reference_torch))

    def state_outcome(self, module):
        compiler = module.compiler
        function = compiler.cudagraph_mark_step_begin
        original_backend = compiler.get_default_backend()
        original_guard_collectives = compiler.set_enable_guard_collectives(False)

        def backend(graph_module, example_inputs):
            return graph_module.forward

        try:
            compiler.set_default_backend(backend)
            results = []
            for context in (contextlib.nullcontext(), module.no_grad()):
                with context:
                    compiler.set_enable_guard_collectives(True)
                    before_grad = module.is_grad_enabled()
                    before_queries = (
                        compiler.is_compiling(),
                        compiler.is_dynamo_compiling(),
                        compiler.is_exporting(),
                    )
                    result = function()
                    after_queries = (
                        compiler.is_compiling(),
                        compiler.is_dynamo_compiling(),
                        compiler.is_exporting(),
                    )
                    previous_guard_collectives = compiler.set_enable_guard_collectives(
                        False
                    )
                    results.append(
                        (
                            result is None,
                            before_grad,
                            module.is_grad_enabled(),
                            before_queries,
                            after_queries,
                            compiler.get_default_backend() is backend,
                            previous_guard_collectives,
                        )
                    )
            return results
        finally:
            compiler.set_default_backend(original_backend)
            compiler.set_enable_guard_collectives(original_guard_collectives)

    def test_imports_exports_copying_and_pickling_match_pytorch_2_13(self):
        actual_compiler = torch.compiler
        expected_compiler = reference_torch.compiler
        actual = actual_compiler.cudagraph_mark_step_begin
        expected = expected_compiler.cudagraph_mark_step_begin

        self.assertEqual(
            actual_compiler.__all__,
            [
                name
                for name in expected_compiler.__all__
                if name in SUPPORTED_COMPILER_EXPORTS
            ],
        )
        self.assertEqual(
            actual_compiler.__all__.count("cudagraph_mark_step_begin"),
            expected_compiler.__all__.count("cudagraph_mark_step_begin"),
        )
        self.assertEqual(
            torch.__all__.count("compiler"),
            reference_torch.__all__.count("compiler"),
        )
        self.assertEqual(
            torch.__all__.count("cudagraph_mark_step_begin"),
            reference_torch.__all__.count("cudagraph_mark_step_begin"),
        )

        for module, function in (
            (actual_compiler, actual),
            (expected_compiler, expected),
        ):
            direct_namespace = {}
            exec(
                f"from {module.__name__} import cudagraph_mark_step_begin",
                direct_namespace,
            )
            self.assertIs(direct_namespace["cudagraph_mark_step_begin"], function)

            wildcard_namespace = {}
            exec(f"from {module.__name__} import *", wildcard_namespace)
            for name in SUPPORTED_COMPILER_EXPORTS:
                self.assertIs(wildcard_namespace[name], getattr(module, name))

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("compiler", namespace)
            self.assertNotIn("cudagraph_mark_step_begin", namespace)

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

    def reload_outcome(self, module, compiler_module_name):
        compiler = importlib.import_module(compiler_module_name)
        function = compiler.cudagraph_mark_step_begin
        original_backend = compiler.get_default_backend()
        original_guard_collectives = compiler.set_enable_guard_collectives(False)

        def backend(graph_module, example_inputs):
            return graph_module.forward

        try:
            compiler.set_default_backend(backend)
            compiler.set_enable_guard_collectives(True)
            old_exports = compiler.__all__
            reloaded = importlib.reload(compiler)
            new_function = reloaded.cudagraph_mark_step_begin

            try:
                pickle.dumps(function)
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
            previous_guard_collectives = compiler.set_enable_guard_collectives(False)
            return (
                reloaded is compiler,
                module.compiler is compiler,
                function is new_function,
                old_exports is compiler.__all__,
                compiler.__all__.count("cudagraph_mark_step_begin"),
                compiler.get_default_backend() is backend,
                new_function() is None,
                previous_guard_collectives,
                copy.copy(function) is function,
                copy.deepcopy(function) is function,
                old_pickle_error,
                new_pickle_results,
            )
        finally:
            compiler.set_default_backend(original_backend)
            compiler.set_enable_guard_collectives(original_guard_collectives)

    def test_reload_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_outcome(torch, "torch_rs.compiler"),
            self.reload_outcome(reference_torch, "torch.compiler"),
        )

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.compiler.cudagraph_mark_step_begin
        expected = reference_torch.compiler.cudagraph_mark_step_begin
        cases = (
            (lambda: actual(None), lambda: expected(None)),
            (lambda: actual(None, None), lambda: expected(None, None)),
            (lambda: actual(enabled=True), lambda: expected(enabled=True)),
            (
                lambda: actual(None, enabled=True),
                lambda: expected(None, enabled=True),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_compile_cache_graph_and_cuda_execution_remain_deliberately_unsupported(
        self,
    ):
        self.assertTrue(callable(reference_torch.compile))
        self.assertTrue(hasattr(reference_torch, "export"))
        self.assertTrue(hasattr(reference_torch, "cuda"))
        self.assertTrue(hasattr(reference_torch.compiler, "compile"))
        self.assertTrue(hasattr(reference_torch.compiler, "save_cache_artifacts"))
        self.assertTrue(hasattr(reference_torch.compiler, "load_cache_artifacts"))

        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch, "export"))
        self.assertFalse(hasattr(torch, "cuda"))
        self.assertFalse(hasattr(torch.compiler, "compile"))
        self.assertFalse(hasattr(torch.compiler, "save_cache_artifacts"))
        self.assertFalse(hasattr(torch.compiler, "load_cache_artifacts"))
        self.assertFalse(hasattr(torch.compiler, "nested_compile_region"))


if __name__ == "__main__":
    unittest.main()
