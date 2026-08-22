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


class CompilerResetTests(unittest.TestCase):
    def compiler_state(self):
        return (
            torch.is_grad_enabled(),
            torch.compiler.get_default_backend(),
            torch.compiler.is_compiling(),
            torch.compiler.is_dynamo_compiling(),
            torch.compiler.is_exporting(),
        )

    def test_repeatable_no_op_preserves_tensors_storage_and_autograd(self):
        function = torch.compiler.reset
        self.assertEqual(function.__code__.co_names, ())
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            requires_grad=True,
        )
        view = leaf.transpose(0, 1)[1]
        loss = (view * torch.tensor([5.0, 7.0])).sum()
        leaf_pointer = leaf.data_ptr()
        view_pointer = view.data_ptr()
        view_offset = view.storage_offset()
        view_stride = view.stride()

        for _ in range(4):
            self.assertIs(function(), None)

        self.assertEqual(leaf.tolist(), [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(view.tolist(), [2.0, 4.0])
        self.assertEqual(leaf.data_ptr(), leaf_pointer)
        self.assertEqual(view.data_ptr(), view_pointer)
        self.assertEqual(view.storage_offset(), view_offset)
        self.assertEqual(view.stride(), view_stride)
        self.assertTrue(leaf.requires_grad)
        self.assertTrue(view.requires_grad)
        self.assertIsNone(leaf.grad)

        loss.backward()
        self.assertEqual(leaf.grad.tolist(), [[0.0, 5.0], [0.0, 7.0]])

    def test_preserves_grad_mode_and_existing_compiler_queries(self):
        expected_enabled = (True, "inductor", False, False, False)
        expected_disabled = (False, "inductor", False, False, False)

        def assert_reset_preserves_state(expected):
            self.assertEqual(self.compiler_state(), expected)
            self.assertIs(torch.compiler.reset(), None)
            self.assertIs(torch.compiler.reset(), None)
            self.assertEqual(self.compiler_state(), expected)

        assert_reset_preserves_state(expected_enabled)
        with torch.no_grad():
            assert_reset_preserves_state(expected_disabled)
            with torch.no_grad():
                assert_reset_preserves_state(expected_disabled)
            assert_reset_preserves_state(expected_disabled)
        assert_reset_preserves_state(expected_enabled)

    def test_threaded_resets_preserve_tensors_grad_mode_and_queries(self):
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
                    before = self.compiler_state()
                    barrier.wait(timeout=10)
                    first = torch.compiler.reset()
                    second = torch.compiler.reset()
                    after = self.compiler_state()
                    results[index] = (
                        first,
                        second,
                        before,
                        after,
                        tensor.data_ptr() == pointer,
                        tensor.tolist(),
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
        for index, result in enumerate(results):
            expected_grad_state = index % 2 == 0
            expected_state = (
                expected_grad_state,
                "inductor",
                False,
                False,
                False,
            )
            self.assertEqual(
                result,
                (
                    None,
                    None,
                    expected_state,
                    expected_state,
                    True,
                    [float(index), float(index + 1)],
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
        self.assertEqual(
            inspect.cleandoc(function.__doc__),
            inspect.cleandoc(FUNCTION_DOC),
        )
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

        self.assertFalse(hasattr(torch, "reset"))
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
                lambda: function(enabled=True),
                "reset() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: function(None, enabled=True),
                "reset() got an unexpected keyword argument 'enabled'",
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
        for name in ("compile", "save_cache_artifacts", "load_cache_artifacts"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.compiler, name))
                self.assertNotIn(name, torch.compiler.__all__)

    def test_importing_and_calling_does_not_import_pytorch_or_dynamo(self):
        script = r'''
import sys

class RejectCompilerImport:
    def find_spec(self, fullname, path=None, target=None):
        if (
            fullname == "torch"
            or fullname.startswith("torch.")
            or fullname == "torch_rs._dynamo"
            or fullname.startswith("torch_rs._dynamo.")
        ):
            raise RuntimeError(f"compiler import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectCompilerImport())
import torch_rs as torch

modules_before_call = set(sys.modules)
assert torch.compiler.reset() is None
assert torch.compiler.reset() is None
assert set(sys.modules) == modules_before_call
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
assert not any(
    name == "torch_rs._dynamo" or name.startswith("torch_rs._dynamo.")
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
