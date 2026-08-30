import contextlib
import copy
import importlib
import inspect
import pickle
import re
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


class CompilerCudagraphMarkStepBeginTests(unittest.TestCase):
    def test_noop_returns_none_and_preserves_grad_compiler_and_cuda_metadata(self):
        compiler = torch.compiler
        function = compiler.cudagraph_mark_step_begin
        original_backend = compiler.get_default_backend()

        def backend(graph_module, example_inputs):
            return graph_module.forward

        try:
            compiler.set_default_backend(backend)
            expected_queries = (
                compiler.is_compiling(),
                compiler.is_dynamo_compiling(),
                compiler.is_exporting(),
            )
            expected_cuda_metadata = (
                torch.backends.cuda.is_built(),
                torch.backends.cudnn.is_available(),
                torch.version.cuda,
                hasattr(torch, "cuda"),
                "torch_rs.cuda" in sys.modules,
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
                    self.assertEqual(
                        (
                            compiler.is_compiling(),
                            compiler.is_dynamo_compiling(),
                            compiler.is_exporting(),
                        ),
                        expected_queries,
                    )
                    self.assertEqual(
                        (
                            torch.backends.cuda.is_built(),
                            torch.backends.cudnn.is_available(),
                            torch.version.cuda,
                            hasattr(torch, "cuda"),
                            "torch_rs.cuda" in sys.modules,
                        ),
                        expected_cuda_metadata,
                    )

            self.assertEqual(
                expected_cuda_metadata,
                (False, False, None, False, False),
            )
        finally:
            compiler.set_default_backend(original_backend)

    def test_signature_documentation_and_module_identity(self):
        compiler = importlib.import_module("torch_rs.compiler")
        function = compiler.cudagraph_mark_step_begin

        self.assertIs(torch.compiler, compiler)
        self.assertIs(sys.modules["torch_rs.compiler"], compiler)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "()")
        self.assertEqual(inspect.get_annotations(function), {})
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

    def test_imports_wildcards_copy_pickle_and_reload_use_canonical_function(self):
        compiler = torch.compiler
        function = compiler.cudagraph_mark_step_begin

        self.assertEqual(compiler.__all__, COMPILER_EXPORTS)

        direct_import = {}
        compiler_namespace = {}
        top_level_namespace = {}
        exec("from torch_rs.compiler import cudagraph_mark_step_begin", direct_import)
        exec("from torch_rs.compiler import *", compiler_namespace)
        exec("from torch_rs import *", top_level_namespace)

        self.assertIs(direct_import["cudagraph_mark_step_begin"], function)
        self.assertEqual(
            {name for name in compiler_namespace if not name.startswith("__")},
            set(COMPILER_EXPORTS),
        )
        for name in COMPILER_EXPORTS:
            self.assertIs(compiler_namespace[name], getattr(compiler, name))

        self.assertNotIn("compiler", torch.__all__)
        self.assertNotIn("cudagraph_mark_step_begin", torch.__all__)
        self.assertFalse(hasattr(torch, "cudagraph_mark_step_begin"))
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
        with self.assertRaises(pickle.PicklingError) as raised:
            pickle.dumps(old_function)
        message = re.sub(r"0x[0-9a-fA-F]+", "0x...", str(raised.exception))
        self.assertEqual(
            message,
            "Can't pickle <function cudagraph_mark_step_begin at 0x...>: "
            "it's not the same object as "
            "torch_rs.compiler.cudagraph_mark_step_begin",
        )
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            self.assertIs(
                pickle.loads(pickle.dumps(new_function, protocol)),
                new_function,
            )

    def test_argument_errors_match_pytorch_2_13(self):
        function = torch.compiler.cudagraph_mark_step_begin
        cases = (
            (
                lambda: function(None),
                "cudagraph_mark_step_begin() takes 0 positional arguments but "
                "1 was given",
            ),
            (
                lambda: function(None, None),
                "cudagraph_mark_step_begin() takes 0 positional arguments but "
                "2 were given",
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
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_compile_graph_backend_registration_and_cuda_tensors_remain_unsupported(self):
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch, "export"))
        self.assertFalse(hasattr(torch, "cudagraph_mark_step_begin"))
        self.assertFalse(hasattr(torch.compiler, "allow_in_graph"))
        self.assertFalse(hasattr(torch.compiler, "compile"))
        self.assertFalse(hasattr(torch.compiler, "register_backend"))
        self.assertFalse(hasattr(torch.compiler, "substitute_in_graph"))
        self.assertFalse(hasattr(torch.compiler, "load_compiled_function"))
        self.assertFalse(hasattr(torch.compiler, "nested_compile_region"))
        self.assertFalse(hasattr(torch.compiler, "wrap_numpy"))
        self.assertNotIn("torch_rs._inductor", sys.modules)
        self.assertFalse(hasattr(torch, "cuda"))
        self.assertFalse(hasattr(torch.Tensor, "cuda"))
        self.assertFalse(hasattr(torch.Tensor, "to"))
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tensor\(\): device 'cuda:0' is not supported; only 'cpu' is implemented$",
        ):
            torch.tensor([1.0], device="cuda:0")

    def test_importing_and_calling_does_not_import_pytorch_or_runtime_modules(self):
        script = r"""
import os
import sys

class RejectExternalRuntimeImport:
    blocked = {
        "amdsmi",
        "cupy",
        "intel_extension_for_pytorch",
        "nvidia",
        "numpy",
        "pynvml",
        "torch",
    }

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in self.blocked:
            raise RuntimeError(f"external runtime import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectExternalRuntimeImport())
os.environ.update(
    CUDA_VISIBLE_DEVICES="0",
    NVIDIA_VISIBLE_DEVICES="all",
    PYTORCH_NVML_BASED_CUDA_CHECK="1",
)
import torch_rs as torch
from torch_rs.compiler import cudagraph_mark_step_begin

modules_before_call = set(sys.modules)
assert torch.compiler.cudagraph_mark_step_begin is cudagraph_mark_step_begin
assert cudagraph_mark_step_begin() is None
assert torch.backends.cuda.is_built() is False
assert torch.backends.cudnn.is_available() is False
assert torch.version.cuda is None
assert not hasattr(torch, "cuda")
assert set(sys.modules) == modules_before_call
assert not any(
    name.split(".", 1)[0] in RejectExternalRuntimeImport.blocked
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
