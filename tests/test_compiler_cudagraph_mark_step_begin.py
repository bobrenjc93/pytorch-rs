import contextlib
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
    Indicates that a new iteration of inference or training is about to begin.

    CUDA Graphs will free tensors of a prior iteration. A new iteration is started on each invocation of
    torch.compile, so long as there is not a pending backward that has not been called.

    If that heuristic is wrong, such as in the following example, manually mark it with this api.

    .. code-block:: python

        @torch.compile(mode="reduce-overhead")
        def rand_foo():
            return torch.rand([4], device="cuda")


        for _ in range(5):
            torch.compiler.cudagraph_mark_step_begin()
            rand_foo() + rand_foo()

    For more details, see `torch.compiler_cudagraph_trees <https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html>`__  # noqa: B950
    """


COMPILER_EXPORTS = [
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
]


class _CallableBackend:
    def __call__(self, graph_module, example_inputs):
        return graph_module.forward


def _backend_state():
    return (
        torch.backends.mha.get_fastpath_enabled(),
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
        torch.backends.cudnn.enabled,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.benchmark_limit,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.allow_tf32,
    )


class CompilerCudagraphMarkStepBeginTests(unittest.TestCase):
    def test_returns_none_and_preserves_compiler_grad_guard_and_backend_state(self):
        compiler = torch.compiler
        function = compiler.cudagraph_mark_step_begin
        original_backend = compiler.get_default_backend()
        original_guard = compiler.set_enable_guard_collectives(False)
        original_mha = torch.backends.mha.get_fastpath_enabled()
        original_flash = torch.backends.cuda.flash_sdp_enabled()
        original_math = torch.backends.cuda.math_sdp_enabled()
        original_mem_efficient = torch.backends.cuda.mem_efficient_sdp_enabled()
        original_fp16_bf16 = (
            torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
        )
        original_cudnn = torch.backends.cudnn.set_flags()

        def backend(graph_module, example_inputs):
            return graph_module.forward

        try:
            compiler.set_default_backend(backend)
            compiler.set_enable_guard_collectives(True)
            torch.backends.mha.set_fastpath_enabled(False)
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_math_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
            torch.backends.cudnn.set_flags(False, True, 7, True, False)
            expected_backend_state = _backend_state()
            expected_compiler_queries = (
                compiler.is_compiling(),
                compiler.is_dynamo_compiling(),
                compiler.is_exporting(),
            )

            for context, expected_grad_state in (
                (contextlib.nullcontext(), True),
                (torch.no_grad(), False),
            ):
                with context:
                    self.assertIs(torch.is_grad_enabled(), expected_grad_state)
                    self.assertIs(function(), None)
                    self.assertIs(torch.is_grad_enabled(), expected_grad_state)
                    self.assertIs(compiler.get_default_backend(), backend)
                    self.assertEqual(_backend_state(), expected_backend_state)
                    self.assertIs(compiler.set_enable_guard_collectives(True), True)
                    self.assertEqual(
                        (
                            compiler.is_compiling(),
                            compiler.is_dynamo_compiling(),
                            compiler.is_exporting(),
                        ),
                        expected_compiler_queries,
                    )
        finally:
            compiler.set_default_backend(original_backend)
            compiler.set_enable_guard_collectives(original_guard)
            torch.backends.mha.set_fastpath_enabled(original_mha)
            torch.backends.cuda.enable_flash_sdp(original_flash)
            torch.backends.cuda.enable_math_sdp(original_math)
            torch.backends.cuda.enable_mem_efficient_sdp(original_mem_efficient)
            torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(
                original_fp16_bf16
            )
            torch.backends.cudnn.set_flags(*original_cudnn)

    def test_signature_annotations_documentation_and_module_identity(self):
        compiler = importlib.import_module("torch_rs.compiler")
        function = compiler.cudagraph_mark_step_begin

        self.assertIs(torch.compiler, compiler)
        self.assertIs(sys.modules["torch_rs.compiler"], compiler)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "()")
        self.assertEqual(function.__annotations__, {})
        self.assertEqual(typing.get_type_hints(function), {})
        self.assertEqual(function.__name__, "cudagraph_mark_step_begin")
        self.assertEqual(function.__qualname__, "cudagraph_mark_step_begin")
        self.assertEqual(function.__module__, "torch_rs.compiler")
        self.assertIs(inspect.getmodule(function), compiler)
        self.assertEqual(
            inspect.cleandoc(function.__doc__),
            inspect.cleandoc(FUNCTION_DOC),
        )
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_direct_wildcard_reload_copy_and_pickle_use_the_canonical_module(self):
        compiler = torch.compiler
        function = compiler.cudagraph_mark_step_begin

        from torch_rs.compiler import cudagraph_mark_step_begin

        self.assertIs(cudagraph_mark_step_begin, function)
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
        self.assertNotIn("cudagraph_mark_step_begin", torch.__all__)
        self.assertFalse(hasattr(torch, "cudagraph_mark_step_begin"))
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("compiler", top_level_namespace)
        self.assertNotIn("cudagraph_mark_step_begin", top_level_namespace)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.compiler", payload)
                self.assertIs(pickle.loads(payload), function)

        old_function = function
        old_exports = compiler.__all__
        reloaded = importlib.reload(compiler)
        new_function = reloaded.cudagraph_mark_step_begin

        self.assertIs(reloaded, compiler)
        self.assertIs(torch.compiler, compiler)
        self.assertIs(sys.modules["torch_rs.compiler"], compiler)
        self.assertIsNot(new_function, old_function)
        self.assertIsNot(compiler.__all__, old_exports)
        self.assertEqual(compiler.__all__, COMPILER_EXPORTS)
        self.assertIs(new_function(), None)
        self.assertIs(copy.copy(old_function), old_function)
        self.assertIs(copy.deepcopy(old_function), old_function)
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(old_function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(new_function, protocol=protocol)
                self.assertIn(b"torch_rs.compiler", payload)
                self.assertIs(pickle.loads(payload), new_function)

    def test_rejects_arguments_with_pytorch_2_13_errors(self):
        function = torch.compiler.cudagraph_mark_step_begin
        cases = (
            (
                lambda: function(None),
                "cudagraph_mark_step_begin() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: function(None, None),
                "cudagraph_mark_step_begin() takes 0 positional arguments but 2 were given",
            ),
            (
                lambda: function(enabled=True),
                "cudagraph_mark_step_begin() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: function(None, enabled=True),
                "cudagraph_mark_step_begin() got an unexpected keyword argument 'enabled'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_compile_cuda_graph_and_cache_surfaces_remain_unsupported(self):
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch, "export"))
        self.assertFalse(hasattr(torch, "cuda"))
        self.assertFalse(hasattr(torch.compiler, "compile"))
        self.assertFalse(hasattr(torch.compiler, "load_compiled_function"))
        self.assertFalse(hasattr(torch.compiler, "save_cache_artifacts"))
        self.assertFalse(hasattr(torch.compiler, "load_cache_artifacts"))
        self.assertFalse(hasattr(torch.compiler, "nested_compile_region"))

    def test_importing_and_calling_does_not_import_pytorch_or_compiler_backends(self):
        script = r"""
import sys

class RejectCompilerImport:
    def find_spec(self, fullname, path=None, target=None):
        if (
            fullname == "torch"
            or fullname.startswith("torch.")
            or fullname == "torch_rs._dynamo"
            or fullname.startswith("torch_rs._dynamo.")
            or fullname == "torch_rs._inductor"
            or fullname.startswith("torch_rs._inductor.")
            or fullname.startswith("torch_rs.compiler.backends")
            or fullname == "torch_rs.compiler.registry"
            or fullname == "torch_rs.cuda"
            or fullname.startswith("torch_rs.cuda.")
        ):
            raise RuntimeError(f"compiler import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectCompilerImport())
import torch_rs as torch

modules_before_call = set(sys.modules)
assert torch.compiler.cudagraph_mark_step_begin() is None
assert set(sys.modules) == modules_before_call
assert not hasattr(torch, "compile")
assert not hasattr(torch, "export")
assert not hasattr(torch, "cuda")
assert not hasattr(torch.compiler, "compile")
assert not hasattr(torch.compiler, "load_compiled_function")
assert not hasattr(torch.compiler, "save_cache_artifacts")
assert not hasattr(torch.compiler, "load_cache_artifacts")
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
assert not any(
    name == "torch_rs._dynamo"
    or name.startswith("torch_rs._dynamo.")
    or name == "torch_rs._inductor"
    or name.startswith("torch_rs._inductor.")
    or name.startswith("torch_rs.compiler.backends")
    or name == "torch_rs.compiler.registry"
    or name == "torch_rs.cuda"
    or name.startswith("torch_rs.cuda.")
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
