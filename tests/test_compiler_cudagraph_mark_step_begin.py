import copy
import importlib
import inspect
import pickle
import subprocess
import sys
import types
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


class CompilerCudagraphMarkStepBeginTests(unittest.TestCase):
    def setUp(self):
        self.original_backend = torch.compiler.get_default_backend()
        self.original_guard_collectives = torch.compiler.set_enable_guard_collectives(
            False
        )

    def tearDown(self):
        torch.compiler.set_default_backend(self.original_backend)
        torch.compiler.set_enable_guard_collectives(self.original_guard_collectives)

    def test_returns_none_and_preserves_eager_compiler_grad_and_guard_state(self):
        compiler = torch.compiler
        function = compiler.cudagraph_mark_step_begin

        def backend(graph_module, example_inputs):
            return graph_module.forward

        compiler.set_default_backend(backend)

        def assert_noop_preserves_state(expected_grad_state):
            compiler.set_enable_guard_collectives(True)
            before_queries = (
                compiler.is_compiling(),
                compiler.is_dynamo_compiling(),
                compiler.is_exporting(),
            )

            self.assertIs(torch.is_grad_enabled(), expected_grad_state)
            self.assertIs(function(), None)
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)
            self.assertIs(compiler.get_default_backend(), backend)
            self.assertEqual(
                (
                    compiler.is_compiling(),
                    compiler.is_dynamo_compiling(),
                    compiler.is_exporting(),
                ),
                before_queries,
            )
            self.assertIs(compiler.set_enable_guard_collectives(False), True)

        assert_noop_preserves_state(True)
        with torch.no_grad():
            assert_noop_preserves_state(False)
        assert_noop_preserves_state(True)

    def test_signature_documentation_and_module_identity(self):
        compiler = importlib.import_module("torch_rs.compiler")
        function = compiler.cudagraph_mark_step_begin

        self.assertIs(torch.compiler, compiler)
        self.assertIs(sys.modules["torch_rs.compiler"], compiler)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "()")
        self.assertEqual(function.__annotations__, {})
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

    def test_direct_and_wildcard_imports_copy_and_pickle_use_canonical_function(self):
        compiler = torch.compiler
        function = compiler.cudagraph_mark_step_begin

        self.assertEqual(compiler.__all__.count("cudagraph_mark_step_begin"), 1)

        direct_namespace = {}
        exec(
            "from torch_rs.compiler import cudagraph_mark_step_begin",
            direct_namespace,
        )
        self.assertIs(direct_namespace["cudagraph_mark_step_begin"], function)

        wildcard_namespace = {}
        exec("from torch_rs.compiler import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["cudagraph_mark_step_begin"], function)

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

    def test_reload_recreates_function_without_mutating_compiler_state(self):
        compiler = torch.compiler

        def backend(graph_module, example_inputs):
            return graph_module.forward

        compiler.set_default_backend(backend)
        compiler.set_enable_guard_collectives(True)
        old_function = compiler.cudagraph_mark_step_begin
        old_exports = compiler.__all__

        reloaded = importlib.reload(compiler)
        new_function = reloaded.cudagraph_mark_step_begin

        self.assertIs(reloaded, compiler)
        self.assertIs(torch.compiler, compiler)
        self.assertIs(sys.modules["torch_rs.compiler"], compiler)
        self.assertIsNot(new_function, old_function)
        self.assertIsNot(compiler.__all__, old_exports)
        self.assertEqual(compiler.__all__.count("cudagraph_mark_step_begin"), 1)
        self.assertIs(compiler.get_default_backend(), backend)
        self.assertIs(new_function(), None)
        self.assertIs(compiler.set_enable_guard_collectives(False), True)
        self.assertIs(copy.copy(old_function), old_function)
        self.assertIs(copy.deepcopy(old_function), old_function)
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(old_function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(new_function, protocol)),
                    new_function,
                )

    def test_rejects_arguments_with_pytorch_2_13_errors_and_preserves_state(self):
        compiler = torch.compiler
        function = compiler.cudagraph_mark_step_begin

        def backend(graph_module, example_inputs):
            return graph_module.forward

        compiler.set_default_backend(backend)
        cases = (
            (
                lambda: function(None),
                "cudagraph_mark_step_begin() takes 0 positional arguments but 1 "
                "was given",
            ),
            (
                lambda: function(None, None),
                "cudagraph_mark_step_begin() takes 0 positional arguments but 2 "
                "were given",
            ),
            (
                lambda: function(enabled=True),
                "cudagraph_mark_step_begin() got an unexpected keyword argument "
                "'enabled'",
            ),
            (
                lambda: function(None, enabled=True),
                "cudagraph_mark_step_begin() got an unexpected keyword argument "
                "'enabled'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                compiler.set_enable_guard_collectives(True)
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertIs(compiler.get_default_backend(), backend)
                self.assertIs(compiler.set_enable_guard_collectives(False), True)

    def test_compile_cache_graph_and_cuda_execution_surfaces_remain_unsupported(self):
        self.assertIs(torch.compiler.cudagraph_mark_step_begin(), None)
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch, "export"))
        self.assertFalse(hasattr(torch, "cuda"))
        self.assertIs(torch.backends.cuda.is_built(), False)

        for name in (
            "compile",
            "load_compiled_function",
            "save_cache_artifacts",
            "load_cache_artifacts",
            "nested_compile_region",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.compiler, name))

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
        ):
            raise RuntimeError(f"compiler import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectCompilerImport())
import torch_rs as torch

compiler = torch.compiler
modules_before_call = set(sys.modules)
backend = lambda graph_module, example_inputs: graph_module.forward
compiler.set_default_backend(backend)
assert compiler.set_enable_guard_collectives(True) is False
assert compiler.cudagraph_mark_step_begin() is None
assert compiler.get_default_backend() is backend
assert compiler.set_enable_guard_collectives(False) is True
assert compiler.is_compiling() is False
assert compiler.is_dynamo_compiling() is False
assert compiler.is_exporting() is False
assert not hasattr(torch, "compile")
assert not hasattr(torch, "export")
assert not hasattr(torch, "cuda")
assert not hasattr(compiler, "compile")
assert not hasattr(compiler, "save_cache_artifacts")
assert not hasattr(compiler, "load_cache_artifacts")
assert set(sys.modules) == modules_before_call
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
assert not any(
    name == "torch_rs._dynamo"
    or name.startswith("torch_rs._dynamo.")
    or name == "torch_rs._inductor"
    or name.startswith("torch_rs._inductor.")
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
