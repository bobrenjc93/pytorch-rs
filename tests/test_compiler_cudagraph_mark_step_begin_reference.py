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

    def marker_outcome(self, module):
        compiler = module.compiler
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

            states = []
            for context, expected_grad_state in (
                (contextlib.nullcontext(), True),
                (module.no_grad(), False),
            ):
                with context:
                    before = module.is_grad_enabled()
                    result = function()
                    after = module.is_grad_enabled()
                    states.append(
                        (
                            before is expected_grad_state,
                            result is None,
                            after is expected_grad_state,
                            compiler.get_default_backend() is backend,
                            (
                                compiler.is_compiling(),
                                compiler.is_dynamo_compiling(),
                                compiler.is_exporting(),
                            )
                            == expected_queries,
                        )
                    )
            return states
        finally:
            compiler.set_default_backend(original_backend)

    def test_noop_state_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.marker_outcome(torch),
            self.marker_outcome(reference_torch),
        )

    def test_signature_documentation_and_metadata_match_pytorch_2_13(self):
        actual_compiler = importlib.import_module("torch_rs.compiler")
        expected_compiler = importlib.import_module("torch.compiler")
        actual = actual_compiler.cudagraph_mark_step_begin
        expected = expected_compiler.cudagraph_mark_step_begin

        self.assertIs(torch.compiler, actual_compiler)
        self.assertIs(reference_torch.compiler, expected_compiler)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual)),
            str(inspect.signature(expected)),
        )
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(
            typing.get_type_hints(actual),
            typing.get_type_hints(expected),
        )
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

    def test_imports_wildcards_copy_pickle_and_reload_match_pytorch_2_13(self):
        actual_compiler = torch.compiler
        expected_compiler = reference_torch.compiler

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
            torch.__all__.count("cudagraph_mark_step_begin"),
            reference_torch.__all__.count("cudagraph_mark_step_begin"),
        )

        for module in (actual_compiler, expected_compiler):
            namespace = {}
            exec(
                f"from {module.__name__} import cudagraph_mark_step_begin",
                namespace,
            )
            self.assertIs(
                namespace["cudagraph_mark_step_begin"],
                module.cudagraph_mark_step_begin,
            )

            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            for name in SUPPORTED_COMPILER_EXPORTS:
                self.assertIs(namespace[name], getattr(module, name))

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("compiler", namespace)
            self.assertNotIn("cudagraph_mark_step_begin", namespace)

        actual = actual_compiler.cudagraph_mark_step_begin
        expected = expected_compiler.cudagraph_mark_step_begin
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

        for package, compiler_module_name in (
            (torch, "torch_rs.compiler"),
            (reference_torch, "torch.compiler"),
        ):
            with self.subTest(package=package.__name__):
                compiler = importlib.import_module(compiler_module_name)
                old_function = compiler.cudagraph_mark_step_begin
                old_exports = compiler.__all__
                reloaded = importlib.reload(compiler)
                new_function = reloaded.cudagraph_mark_step_begin

                self.assertIs(reloaded, compiler)
                self.assertIs(package.compiler, compiler)
                self.assertIs(sys.modules[compiler_module_name], compiler)
                self.assertIsNot(new_function, old_function)
                self.assertIsNot(compiler.__all__, old_exports)
                self.assertIs(new_function(), None)
                with self.assertRaises(pickle.PicklingError):
                    pickle.dumps(old_function)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    self.assertIs(
                        pickle.loads(pickle.dumps(new_function, protocol)),
                        new_function,
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

    def test_cuda_graph_compile_backend_and_cuda_tensor_boundaries_stay_unsupported(self):
        self.assertTrue(callable(reference_torch.compile))
        self.assertTrue(callable(reference_torch.compiler.compile))
        self.assertTrue(callable(reference_torch.compiler.cudagraph_mark_step_begin))
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

    def test_subprocess_import_does_not_import_real_pytorch(self):
        script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
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
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
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
