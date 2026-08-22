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
    Reset the in-process compiler state.

    This function clears Dynamo's in-memory compilation caches and related
    process-local state used by :func:`torch.compile`. It does not delete
    filesystem caches, such as Inductor's disk cache.
    """


def _compiler_state():
    return (
        torch.compiler.get_default_backend(),
        torch.compiler.is_compiling(),
        torch.compiler.is_dynamo_compiling(),
        torch.compiler.is_exporting(),
    )


class CompilerResetTests(unittest.TestCase):
    def test_repeatable_noop_preserves_tensor_views_and_autograd_graphs(self):
        function = torch.compiler.reset
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        view = leaf.view(4)
        weights = torch.tensor([2.0, -1.0, 0.5, 3.0])
        loss = (view * weights).sum()
        leaf_identity = id(leaf)
        view_identity = id(view)
        leaf_pointer = leaf.data_ptr()
        view_offset = view.storage_offset()
        state = _compiler_state()

        self.assertEqual(function.__code__.co_names, ())
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())
        for _ in range(5):
            self.assertIs(function(), None)
            self.assertEqual(_compiler_state(), state)

        self.assertEqual(id(leaf), leaf_identity)
        self.assertEqual(id(view), view_identity)
        self.assertEqual(leaf.data_ptr(), leaf_pointer)
        self.assertEqual(view.data_ptr(), leaf_pointer)
        self.assertEqual(view.storage_offset(), view_offset)
        self.assertEqual(leaf.tolist(), [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(view.tolist(), [1.0, 2.0, 3.0, 4.0])
        self.assertIsNone(leaf.grad)

        loss.backward()
        self.assertEqual(leaf.grad.tolist(), [[2.0, -1.0], [0.5, 3.0]])

    def test_noop_preserves_grad_mode_and_queries_across_threads(self):
        function = torch.compiler.reset
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = torch.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    tensor = torch.tensor(
                        [float(index), float(index + 1)],
                        requires_grad=True,
                    )
                    pointer = tensor.data_ptr()
                    before = (torch.is_grad_enabled(), _compiler_state())
                    barrier.wait(timeout=10)
                    first = function()
                    middle = (torch.is_grad_enabled(), _compiler_state())
                    second = function()
                    after = (torch.is_grad_enabled(), _compiler_state())
                    results[index] = (
                        before,
                        first is None,
                        middle,
                        second is None,
                        after,
                        tensor.tolist(),
                        tensor.data_ptr() == pointer,
                        tensor.requires_grad,
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
        expected_compiler_state = ("inductor", False, False, False)
        for index, result in enumerate(results):
            expected_grad_state = index % 2 == 0
            expected_state = (expected_grad_state, expected_compiler_state)
            self.assertEqual(
                result,
                (
                    expected_state,
                    True,
                    expected_state,
                    True,
                    expected_state,
                    [float(index), float(index + 1)],
                    True,
                    True,
                ),
            )

    def test_signature_annotations_documentation_and_module_identity(self):
        compiler = importlib.import_module("torch_rs.compiler")
        function = compiler.reset

        self.assertIs(torch.compiler, compiler)
        self.assertIs(sys.modules["torch_rs.compiler"], compiler)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "() -> None")
        self.assertEqual(function.__annotations__, {"return": None})
        self.assertEqual(typing.get_type_hints(function), {"return": type(None)})
        self.assertEqual(function.__name__, "reset")
        self.assertEqual(function.__qualname__, "reset")
        self.assertEqual(function.__module__, "torch_rs.compiler")
        self.assertIs(inspect.getmodule(function), compiler)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_exports_copy_and_pickle_use_the_canonical_module(self):
        compiler = torch.compiler
        function = compiler.reset

        self.assertEqual(
            compiler.__all__,
            [
                "assume_constant_result",
                "reset",
                "get_default_backend",
                "is_compiling",
                "is_dynamo_compiling",
                "is_exporting",
            ],
        )
        compiler_namespace = {}
        exec("from torch_rs.compiler import *", compiler_namespace)
        self.assertEqual(
            {name for name in compiler_namespace if not name.startswith("__")},
            set(compiler.__all__),
        )
        for name in compiler.__all__:
            self.assertIs(compiler_namespace[name], getattr(compiler, name))

        self.assertNotIn("compiler", torch.__all__)
        self.assertNotIn("reset", torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("compiler", top_level_namespace)
        self.assertNotIn("reset", top_level_namespace)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.compiler", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_rejects_arguments_with_pytorch_2_13_errors(self):
        function = torch.compiler.reset
        cases = (
            (
                lambda: function(None),
                "reset() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: function(None, None),
                "reset() takes 0 positional arguments but 2 were given",
            ),
            (
                lambda: function(state=None),
                "reset() got an unexpected keyword argument 'state'",
            ),
            (
                lambda: function(None, state=None),
                "reset() got an unexpected keyword argument 'state'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_compilation_and_cache_serialization_remain_unsupported(self):
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch, "export"))
        for name in (
            "compile",
            "load_compiled_function",
            "save_cache_artifacts",
            "load_cache_artifacts",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.compiler, name))
                self.assertNotIn(name, torch.compiler.__all__)

    def test_importing_and_resetting_do_not_import_pytorch_or_dynamo(self):
        script = r"""
import sys

class RejectCompilerImports:
    def find_spec(self, fullname, path=None, target=None):
        if (
            fullname == "torch"
            or fullname.startswith("torch.")
            or fullname == "torch_rs._dynamo"
            or fullname.startswith("torch_rs._dynamo.")
            or fullname == "torch_rs.dynamo"
            or fullname.startswith("torch_rs.dynamo.")
        ):
            raise RuntimeError(f"compiler import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectCompilerImports())
import torch_rs as torch

tensor = torch.tensor([1.0, 2.0], requires_grad=True)
pointer = tensor.data_ptr()
modules_before_call = set(sys.modules)
assert torch.compiler.reset() is None
assert torch.compiler.reset() is None
assert set(sys.modules) == modules_before_call
assert tensor.data_ptr() == pointer
assert tensor.tolist() == [1.0, 2.0]
assert tensor.requires_grad is True
assert torch.compiler.get_default_backend() == "inductor"
assert torch.compiler.is_compiling() is False
assert torch.compiler.is_dynamo_compiling() is False
assert torch.compiler.is_exporting() is False
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
assert not any(
    name == "torch_rs._dynamo"
    or name.startswith("torch_rs._dynamo.")
    or name == "torch_rs.dynamo"
    or name.startswith("torch_rs.dynamo.")
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
