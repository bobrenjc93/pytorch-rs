import contextlib
import copy
import importlib
import inspect
import pickle
import subprocess
import sys
import threading
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
    "cudagraph_mark_step_begin",
    "is_compiling",
    "is_dynamo_compiling",
    "is_exporting",
    "keep_portable_guards_unsafe",
    "skip_guard_on_inbuilt_nn_modules_unsafe",
    "skip_guard_on_all_nn_modules_unsafe",
    "skip_guard_on_globals_unsafe",
    "skip_all_guards_unsafe",
]


class CompilerCudagraphMarkStepBeginTests(unittest.TestCase):
    def setUp(self):
        self.original_backend = torch.compiler.get_default_backend()

    def tearDown(self):
        torch.compiler.set_default_backend(self.original_backend)

    def test_repeated_eager_calls_are_probe_free_and_preserve_state(self):
        compiler = torch.compiler
        function = compiler.cudagraph_mark_step_begin

        def backend(graph_module, example_inputs):
            return graph_module.forward

        compiler.set_default_backend(backend)
        expected_compiler_state = (
            compiler.is_compiling(),
            compiler.is_dynamo_compiling(),
            compiler.is_exporting(),
        )
        modules_before = set(sys.modules)

        for context, expected_grad_state in (
            (contextlib.nullcontext(), True),
            (torch.no_grad(), False),
        ):
            with context:
                for _ in range(5):
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
                        expected_compiler_state,
                    )

        self.assertEqual(set(sys.modules), modules_before)
        self.assertEqual(function.__code__.co_names, ())
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

    def test_threaded_calls_preserve_grad_and_compiler_state(self):
        compiler = torch.compiler
        function = compiler.cudagraph_mark_step_begin
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def backend(graph_module, example_inputs):
            return graph_module.forward

        compiler.set_default_backend(backend)
        expected_compiler_state = (
            compiler.is_compiling(),
            compiler.is_dynamo_compiling(),
            compiler.is_exporting(),
        )

        def worker(index):
            try:
                context = torch.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    before = torch.is_grad_enabled()
                    first = function()
                    middle = torch.is_grad_enabled()
                    second = function()
                    results[index] = (
                        before,
                        first,
                        middle,
                        second,
                        torch.is_grad_enabled(),
                        compiler.get_default_backend() is backend,
                        (
                            compiler.is_compiling(),
                            compiler.is_dynamo_compiling(),
                            compiler.is_exporting(),
                        ),
                    )
            except BaseException as error:
                errors.append(error)

        threads = [
            threading.Thread(target=worker, args=(index,))
            for index in range(worker_count)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        for index, result in enumerate(results):
            expected_grad_state = index % 2 == 0
            self.assertEqual(
                result,
                (
                    expected_grad_state,
                    None,
                    expected_grad_state,
                    None,
                    expected_grad_state,
                    True,
                    expected_compiler_state,
                ),
            )

    def test_signature_metadata_and_module_identity_match_pytorch_2_13(self):
        compiler = importlib.import_module("torch_rs.compiler")
        function = compiler.cudagraph_mark_step_begin

        self.assertIs(torch.compiler, compiler)
        self.assertIs(sys.modules["torch_rs.compiler"], compiler)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(inspect.signature(function), inspect.Signature())
        self.assertEqual(function.__annotations__, {})
        self.assertEqual(typing.get_type_hints(function), {})
        self.assertEqual(function.__name__, "cudagraph_mark_step_begin")
        self.assertEqual(function.__qualname__, "cudagraph_mark_step_begin")
        self.assertEqual(function.__module__, "torch_rs.compiler")
        self.assertIs(inspect.getmodule(function), compiler)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_exports_copying_and_pickling_use_the_canonical_function(self):
        compiler = torch.compiler
        function = compiler.cudagraph_mark_step_begin

        self.assertEqual(compiler.__all__, COMPILER_EXPORTS)
        self.assertEqual(compiler.__all__.count(function.__name__), 1)
        namespace = {}
        exec("from torch_rs.compiler import *", namespace)
        self.assertEqual(
            {name for name in namespace if not name.startswith("__")},
            set(COMPILER_EXPORTS),
        )
        for name in COMPILER_EXPORTS:
            self.assertIs(namespace[name], getattr(compiler, name))

        explicit_namespace = {}
        exec(
            "from torch_rs.compiler import cudagraph_mark_step_begin",
            explicit_namespace,
        )
        self.assertIs(explicit_namespace[function.__name__], function)

        self.assertNotIn("compiler", torch.__all__)
        self.assertNotIn(function.__name__, torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("compiler", top_level_namespace)
        self.assertNotIn(function.__name__, top_level_namespace)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.compiler", payload)
                self.assertIs(pickle.loads(payload), function)

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
                lambda: function(step=None),
                "cudagraph_mark_step_begin() got an unexpected keyword argument 'step'",
            ),
            (
                lambda: function(None, step=None),
                "cudagraph_mark_step_begin() got an unexpected keyword argument 'step'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_reload_replaces_the_function_and_preserves_compiler_state(self):
        compiler = torch.compiler
        old_function = compiler.cudagraph_mark_step_begin
        old_exports = compiler.__all__

        def backend(graph_module, example_inputs):
            return graph_module.forward

        compiler.set_default_backend(backend)
        self.assertIs(old_function(), None)
        reloaded = importlib.reload(compiler)
        new_function = reloaded.cudagraph_mark_step_begin

        self.assertIs(reloaded, compiler)
        self.assertIs(torch.compiler, compiler)
        self.assertIs(sys.modules["torch_rs.compiler"], compiler)
        self.assertIsNot(new_function, old_function)
        self.assertIsNot(compiler.__all__, old_exports)
        self.assertEqual(compiler.__all__, COMPILER_EXPORTS)
        self.assertIs(compiler.get_default_backend(), backend)
        self.assertIs(old_function(), None)
        self.assertIs(new_function(), None)
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(old_function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            self.assertIs(
                pickle.loads(pickle.dumps(new_function, protocol)),
                new_function,
            )

    def test_compilation_remains_unsupported(self):
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch.compiler, "compile"))
        self.assertFalse(hasattr(torch.compiler, "config"))

    def test_importing_and_calling_does_not_probe_pytorch_or_compilers(self):
        script = r'''
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
            or fullname == "torch_rs.compiler.config"
            or fullname == "torch_rs.compiler.registry"
        ):
            raise RuntimeError(f"compiler import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectCompilerImport())
import torch_rs as torch

function = torch.compiler.cudagraph_mark_step_begin
modules_before_call = set(sys.modules)
for _ in range(10):
    assert function() is None
assert set(sys.modules) == modules_before_call
assert function.__code__.co_names == ()
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
assert not any(
    name == "torch_rs._dynamo"
    or name.startswith("torch_rs._dynamo.")
    or name == "torch_rs._inductor"
    or name.startswith("torch_rs._inductor.")
    or name.startswith("torch_rs.compiler.backends")
    or name == "torch_rs.compiler.config"
    or name == "torch_rs.compiler.registry"
    for name in sys.modules
)
'''
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
