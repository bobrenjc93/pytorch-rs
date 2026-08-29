import ast
import contextlib
import copy
import importlib
import inspect
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


class _CallableBackend:
    def __call__(self, graph_module, example_inputs):
        return graph_module.forward


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

    def backend_state(self, module):
        return (
            module.backends.mha.get_fastpath_enabled(),
            module.backends.cuda.flash_sdp_enabled(),
            module.backends.cuda.math_sdp_enabled(),
            module.backends.cuda.mem_efficient_sdp_enabled(),
            module.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
            module.backends.cudnn.enabled,
            module.backends.cudnn.benchmark,
            module.backends.cudnn.benchmark_limit,
            module.backends.cudnn.deterministic,
            module.backends.cudnn.allow_tf32,
        )

    def no_op_state_outcome(self, module):
        compiler = module.compiler
        function = compiler.cudagraph_mark_step_begin
        original_backend = compiler.get_default_backend()
        original_guard = compiler.set_enable_guard_collectives(False)
        original_mha = module.backends.mha.get_fastpath_enabled()
        original_flash = module.backends.cuda.flash_sdp_enabled()
        original_math = module.backends.cuda.math_sdp_enabled()
        original_mem_efficient = module.backends.cuda.mem_efficient_sdp_enabled()
        original_fp16_bf16 = (
            module.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
        )
        original_cudnn = module.backends.cudnn.set_flags()

        backend = _CallableBackend()
        try:
            compiler.set_default_backend(backend)
            compiler.set_enable_guard_collectives(True)
            module.backends.mha.set_fastpath_enabled(False)
            module.backends.cuda.enable_flash_sdp(False)
            module.backends.cuda.enable_math_sdp(False)
            module.backends.cuda.enable_mem_efficient_sdp(False)
            module.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
            module.backends.cudnn.set_flags(False, True, 7, True, False)
            expected_backend_state = self.backend_state(module)
            expected_compiler_queries = (
                compiler.is_compiling(),
                compiler.is_dynamo_compiling(),
                compiler.is_exporting(),
            )

            outcomes = []
            for context in (contextlib.nullcontext(), module.no_grad()):
                with context:
                    before_grad = module.is_grad_enabled()
                    result = function()
                    after_grad = module.is_grad_enabled()
                    previous_guard = compiler.set_enable_guard_collectives(True)
                    outcomes.append(
                        (
                            before_grad,
                            result is None,
                            after_grad,
                            compiler.get_default_backend() is backend,
                            previous_guard,
                            self.backend_state(module),
                            (
                                compiler.is_compiling(),
                                compiler.is_dynamo_compiling(),
                                compiler.is_exporting(),
                            ),
                        )
                    )
            return outcomes, expected_backend_state, expected_compiler_queries
        finally:
            compiler.set_default_backend(original_backend)
            compiler.set_enable_guard_collectives(original_guard)
            module.backends.mha.set_fastpath_enabled(original_mha)
            module.backends.cuda.enable_flash_sdp(original_flash)
            module.backends.cuda.enable_math_sdp(original_math)
            module.backends.cuda.enable_mem_efficient_sdp(original_mem_efficient)
            module.backends.cuda.allow_fp16_bf16_reduction_math_sdp(
                original_fp16_bf16
            )
            module.backends.cudnn.set_flags(*original_cudnn)

    def test_no_op_state_effects_match_pytorch_2_13(self):
        self.assertEqual(
            self.no_op_state_outcome(torch),
            self.no_op_state_outcome(reference_torch),
        )

    def test_signature_annotations_documentation_and_identity_match(self):
        actual_compiler = importlib.import_module("torch_rs.compiler")
        expected_compiler = importlib.import_module("torch.compiler")
        actual = actual_compiler.cudagraph_mark_step_begin
        expected = expected_compiler.cudagraph_mark_step_begin

        self.assertIs(torch.compiler, actual_compiler)
        self.assertIs(reference_torch.compiler, expected_compiler)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(str(inspect.signature(actual)), str(inspect.signature(expected)))
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

    def test_exports_direct_wildcard_copy_and_pickle_match_pytorch_2_13(self):
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
            self.assertIs(wildcard_namespace["cudagraph_mark_step_begin"], function)

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("cudagraph_mark_step_begin", namespace)

        self.assertIs(copy.copy(actual), actual)
        self.assertIs(copy.copy(expected), expected)
        self.assertIs(copy.deepcopy(actual), actual)
        self.assertIs(copy.deepcopy(expected), expected)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(pickle.loads(pickle.dumps(expected, protocol)), expected)
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

    def reload_outcome(self, package_name):
        script = r"""
import importlib
import pickle
import sys

package = importlib.import_module(sys.argv[1])
module_name = f"{package.__name__}.compiler"
original_module = package.compiler
old_function = original_module.cudagraph_mark_step_begin
old_exports = original_module.__all__
reloaded = importlib.reload(original_module)
new_function = reloaded.cudagraph_mark_step_begin
try:
    pickle.dumps(old_function)
except Exception as error:
    old_pickle_outcome = (type(error).__name__, "not the same object" in str(error))
else:
    old_pickle_outcome = None

sys.modules.pop(module_name)
replacement_module = importlib.import_module(module_name)
print(
    repr(
        (
            (
                reloaded is original_module,
                package.compiler is original_module,
                new_function is old_function,
                reloaded.__all__ is old_exports,
                new_function() is None,
                old_pickle_outcome,
            ),
            (
                replacement_module is original_module,
                package.compiler is replacement_module,
                replacement_module.cudagraph_mark_step_begin is new_function,
                replacement_module.cudagraph_mark_step_begin() is None,
            ),
        )
    )
)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script, package_name],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )
        return ast.literal_eval(completed.stdout.strip())

    def test_reload_and_reimport_match_pytorch_2_13(self):
        self.assertEqual(
            self.reload_outcome("torch_rs"),
            self.reload_outcome("torch"),
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

    def test_compilation_cuda_graphs_and_cache_execution_remain_unsupported(self):
        self.assertTrue(callable(reference_torch.compile))
        self.assertTrue(hasattr(reference_torch, "export"))
        self.assertTrue(hasattr(reference_torch, "cuda"))
        self.assertTrue(
            hasattr(reference_torch.compiler, "cudagraph_mark_step_begin")
        )
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch, "export"))
        self.assertFalse(hasattr(torch, "cuda"))
        self.assertFalse(hasattr(torch.compiler, "compile"))
        self.assertFalse(hasattr(torch.compiler, "load_compiled_function"))
        self.assertFalse(hasattr(torch.compiler, "save_cache_artifacts"))
        self.assertFalse(hasattr(torch.compiler, "load_cache_artifacts"))
        self.assertFalse(hasattr(torch.compiler, "nested_compile_region"))


if __name__ == "__main__":
    unittest.main()
