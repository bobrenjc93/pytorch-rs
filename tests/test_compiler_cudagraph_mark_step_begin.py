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
]


class _CallableBackend:
    def __call__(self, graph_module, example_inputs):
        return graph_module.forward


class CompilerCudagraphMarkStepBeginTests(unittest.TestCase):
    def test_no_op_returns_none_and_preserves_grad_compiler_and_cuda_state(self):
        compiler = torch.compiler
        function = compiler.cudagraph_mark_step_begin
        original_backend = compiler.get_default_backend()
        original_guard_collectives = compiler.set_enable_guard_collectives(False)
        backend = _CallableBackend()

        try:
            self.assertIs(compiler.set_default_backend(backend), None)
            self.assertIs(compiler.set_enable_guard_collectives(True), False)

            def assert_preserves_state(expected_grad_state):
                cuda_available = torch.cuda.is_available()
                cuda_initialized = torch.cuda.is_initialized()

                self.assertIs(torch.is_grad_enabled(), expected_grad_state)
                self.assertIs(compiler.get_default_backend(), backend)
                self.assertEqual(
                    (
                        compiler.is_compiling(),
                        compiler.is_dynamo_compiling(),
                        compiler.is_exporting(),
                    ),
                    (False, False, False),
                )
                self.assertIs(compiler.set_enable_guard_collectives(True), True)

                self.assertIs(function(), None)

                self.assertIs(torch.is_grad_enabled(), expected_grad_state)
                self.assertIs(compiler.get_default_backend(), backend)
                self.assertEqual(
                    (
                        compiler.is_compiling(),
                        compiler.is_dynamo_compiling(),
                        compiler.is_exporting(),
                    ),
                    (False, False, False),
                )
                self.assertIs(compiler.set_enable_guard_collectives(True), True)
                self.assertEqual(torch.cuda.is_available(), cuda_available)
                self.assertIs(type(torch.cuda.is_available()), bool)
                self.assertEqual(torch.cuda.is_initialized(), cuda_initialized)
                self.assertIs(type(torch.cuda.is_initialized()), bool)

            assert_preserves_state(True)
            with torch.no_grad():
                assert_preserves_state(False)
                with torch.no_grad():
                    assert_preserves_state(False)
            assert_preserves_state(True)
        finally:
            compiler.set_default_backend(original_backend)
            compiler.set_enable_guard_collectives(original_guard_collectives)

    def test_signature_imports_exports_copy_pickle_and_reload(self):
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
            inspect.cleandoc(function.__doc__), inspect.cleandoc(FUNCTION_DOC)
        )
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

        self.assertEqual(compiler.__all__, COMPILER_EXPORTS)
        package_import = {}
        direct_import = {}
        module_wildcard = {}
        top_level_wildcard = {}
        exec("from torch_rs import compiler", package_import)
        exec(
            "from torch_rs.compiler import cudagraph_mark_step_begin",
            direct_import,
        )
        exec("from torch_rs.compiler import *", module_wildcard)
        exec("from torch_rs import *", top_level_wildcard)
        self.assertIs(package_import["compiler"], compiler)
        self.assertIs(direct_import["cudagraph_mark_step_begin"], function)
        self.assertEqual(
            {name for name in module_wildcard if not name.startswith("__")},
            set(COMPILER_EXPORTS),
        )
        for name in COMPILER_EXPORTS:
            self.assertIs(module_wildcard[name], getattr(compiler, name))
        self.assertNotIn("compiler", torch.__all__)
        self.assertNotIn("cudagraph_mark_step_begin", torch.__all__)
        self.assertNotIn("compiler", top_level_wildcard)
        self.assertNotIn("cudagraph_mark_step_begin", top_level_wildcard)

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
        self.assertIs(copy.copy(new_function), new_function)
        self.assertIs(copy.deepcopy(new_function), new_function)
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(old_function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(reloaded_protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(new_function, protocol)),
                    new_function,
                )

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

    def test_subprocess_call_is_isolated_from_pytorch_compilers_and_cuda_runtime(self):
        script = r"""
import importlib
import pickle
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

compiler = torch.compiler

def backend(graph_module, example_inputs):
    return graph_module.forward

original_guard_collectives = compiler.set_enable_guard_collectives(False)
compiler.set_default_backend(backend)
assert compiler.set_enable_guard_collectives(True) is False

modules_before_call = set(sys.modules)
grad_before = torch.is_grad_enabled()
cuda_available = torch.cuda.is_available()
cuda_initialized = torch.cuda.is_initialized()
assert compiler.cudagraph_mark_step_begin() is None
assert set(sys.modules) == modules_before_call
assert torch.is_grad_enabled() is grad_before
assert compiler.get_default_backend() is backend
assert compiler.set_enable_guard_collectives(True) is True
assert torch.cuda.is_available() is cuda_available
assert torch.cuda.is_initialized() is cuda_initialized
assert compiler.is_compiling() is False
assert compiler.is_dynamo_compiling() is False
assert compiler.is_exporting() is False
assert pickle.loads(pickle.dumps(compiler.cudagraph_mark_step_begin)) is compiler.cudagraph_mark_step_begin
assert importlib.reload(compiler) is compiler
assert compiler.cudagraph_mark_step_begin() is None
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
assert not any(
    name.startswith("torch_rs._dynamo")
    or name.startswith("torch_rs.compiler.backends")
    or name == "torch_rs.compiler.registry"
    for name in sys.modules
)
compiler.set_enable_guard_collectives(original_guard_collectives)
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

    def test_compile_graph_capture_cuda_graphs_and_performance_remain_unsupported(self):
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch, "export"))
        self.assertFalse(hasattr(torch, "is_compiling"))
        self.assertFalse(hasattr(torch, "cudagraph_mark_step_begin"))

        for name in (
            "compile",
            "config",
            "allow_in_graph",
            "substitute_in_graph",
            "set_stance",
            "load_compiled_function",
            "wrap_numpy",
            "save_cache_artifacts",
            "load_cache_artifacts",
            "nested_compile_region",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.compiler, name))

        for name in (
            "CUDAGraph",
            "graph",
            "graph_pool_handle",
            "make_graphed_callables",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.cuda, name))
        self.assertIs(torch.cuda.is_available(), False)
        self.assertEqual(torch.cuda.device_count(), 0)
        self.assertIs(torch.cuda.is_initialized(), False)


if __name__ == "__main__":
    unittest.main()
