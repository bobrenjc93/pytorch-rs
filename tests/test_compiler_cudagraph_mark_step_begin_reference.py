import contextlib
import copy
import importlib
import inspect
import pickle
import pickletools
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


SUPPORTED_COMPILER_EXPORTS = {
    "assume_constant_result",
    "reset",
    "list_backends",
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


class _CallableBackend:
    def __call__(self, graph_module, example_inputs):
        return graph_module.forward


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CompilerCudagraphMarkStepBeginReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "compiler.cudagraph_mark_step_begin differentials require pinned "
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

    def state_observation(self, module):
        compiler = module.compiler
        function = compiler.cudagraph_mark_step_begin
        original_backend = compiler.get_default_backend()
        original_guard_collectives = compiler.set_enable_guard_collectives(False)
        backend = _CallableBackend()
        observations = []

        try:
            compiler.set_default_backend(backend)
            observations.append(compiler.set_enable_guard_collectives(True))
            for context, expected_grad_state in (
                (contextlib.nullcontext(), True),
                (module.no_grad(), False),
            ):
                with context:
                    cuda_available = module.cuda.is_available()
                    cuda_initialized = module.cuda.is_initialized()
                    before_guard_collectives = compiler.set_enable_guard_collectives(
                        True
                    )
                    result = function()
                    after_guard_collectives = compiler.set_enable_guard_collectives(
                        True
                    )
                    observations.append(
                        (
                            module.is_grad_enabled(),
                            expected_grad_state,
                            result is None,
                            compiler.get_default_backend() is backend,
                            before_guard_collectives,
                            after_guard_collectives,
                            module.cuda.is_available() == cuda_available,
                            type(module.cuda.is_available()) is bool,
                            module.cuda.is_initialized() == cuda_initialized,
                            type(module.cuda.is_initialized()) is bool,
                            (
                                compiler.is_compiling(),
                                compiler.is_dynamo_compiling(),
                                compiler.is_exporting(),
                            ),
                        )
                    )
        finally:
            compiler.set_default_backend(original_backend)
            compiler.set_enable_guard_collectives(original_guard_collectives)

        return observations

    def reload_outcome(self, module):
        compiler = module.compiler
        old_function = compiler.cudagraph_mark_step_begin
        old_exports = compiler.__all__
        reloaded = importlib.reload(compiler)
        new_function = reloaded.cudagraph_mark_step_begin
        try:
            pickle.dumps(old_function)
        except pickle.PicklingError:
            old_pickles = False
        else:
            old_pickles = True
        return (
            reloaded is compiler,
            module.compiler is compiler,
            new_function is old_function,
            compiler.__all__ is old_exports,
            "cudagraph_mark_step_begin" in compiler.__all__,
            new_function() is None,
            old_pickles,
            all(
                pickle.loads(pickle.dumps(new_function, protocol)) is new_function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        )

    def test_no_op_and_state_preservation_match_pytorch_2_13(self):
        self.assertEqual(
            self.state_observation(torch),
            self.state_observation(reference_torch),
        )

    def test_signature_metadata_imports_exports_copy_and_pickle_match_pytorch_2_13(
        self,
    ):
        actual_compiler = importlib.import_module("torch_rs.compiler")
        expected_compiler = importlib.import_module("torch.compiler")
        actual = actual_compiler.cudagraph_mark_step_begin
        expected = expected_compiler.cudagraph_mark_step_begin

        self.assertIs(torch.compiler, actual_compiler)
        self.assertIs(reference_torch.compiler, expected_compiler)
        self.assertIs(type(actual), type(expected))
        self.assertEqual(
            str(inspect.signature(actual)), str(inspect.signature(expected))
        )
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertIs(inspect.getmodule(actual), actual_compiler)
        self.assertIs(inspect.getmodule(expected), expected_compiler)
        self.assertEqual(
            inspect.cleandoc(actual.__doc__),
            inspect.cleandoc(expected.__doc__),
        )
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )

        self.assertEqual(
            actual_compiler.__all__,
            [
                name
                for name in expected_compiler.__all__
                if name in SUPPORTED_COMPILER_EXPORTS
            ],
        )
        actual_namespace = {}
        expected_namespace = {}
        exec("from torch_rs.compiler import *", actual_namespace)
        exec("from torch.compiler import *", expected_namespace)
        self.assertEqual(
            {name for name in actual_namespace if not name.startswith("__")},
            {
                name
                for name in expected_namespace
                if name in SUPPORTED_COMPILER_EXPORTS
            },
        )
        self.assertIs(actual_namespace["cudagraph_mark_step_begin"], actual)
        self.assertIs(expected_namespace["cudagraph_mark_step_begin"], expected)
        self.assertEqual(
            torch.__all__.count("cudagraph_mark_step_begin"),
            reference_torch.__all__.count("cudagraph_mark_step_begin"),
        )
        self.assertEqual(
            torch.__all__.count("compiler"),
            reference_torch.__all__.count("compiler"),
        )

        self.assertIs(copy.copy(actual), actual)
        self.assertIs(copy.deepcopy(actual), actual)
        self.assertIs(copy.copy(expected), expected)
        self.assertIs(copy.deepcopy(expected), expected)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(pickle.loads(pickle.dumps(expected, protocol)), expected)

    def test_reload_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_outcome(torch),
            self.reload_outcome(reference_torch),
        )

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.compiler.cudagraph_mark_step_begin
        expected = reference_torch.compiler.cudagraph_mark_step_begin
        cases = (
            (lambda function: function(None),),
            (lambda function: function(None, None),),
            (lambda function: function(enabled=True),),
            (lambda function: function(None, enabled=True),),
        )
        for (call,) in cases:
            with self.subTest(call=call):
                self.assert_error_matches(
                    lambda call=call: call(actual),
                    lambda call=call: call(expected),
                )

    def test_compile_cuda_graph_and_backend_surfaces_remain_unsupported(self):
        self.assertTrue(callable(reference_torch.compile))
        self.assertTrue(hasattr(reference_torch.cuda, "CUDAGraph"))
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch, "export"))
        self.assertFalse(hasattr(torch.compiler, "compile"))
        self.assertFalse(hasattr(torch.compiler, "allow_in_graph"))
        self.assertFalse(hasattr(torch.cuda, "CUDAGraph"))
        self.assertFalse(hasattr(torch.cuda, "graph"))


if __name__ == "__main__":
    unittest.main()
