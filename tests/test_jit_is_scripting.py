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
    Function that returns True when in compilation and False otherwise. This
    is useful especially with the @unused decorator to leave code in your
    model that is not yet TorchScript compatible.
    .. testcode::

        import torch

        @torch.jit.unused
        def unsupported_linear_op(x):
            return x

        def linear(x):
            if torch.jit.is_scripting():
                return torch.linear(x)
            else:
                return unsupported_linear_op(x)
    """


class JitIsScriptingTests(unittest.TestCase):
    def test_eager_false_is_exact_and_preserves_grad_mode(self):
        function = torch.jit.is_scripting

        def assert_query_preserves_grad_mode(expected_grad_state):
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)
            self.assertIs(function(), False)
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)

        assert_query_preserves_grad_mode(True)
        with torch.no_grad():
            assert_query_preserves_grad_mode(False)
            with torch.no_grad():
                assert_query_preserves_grad_mode(False)
            assert_query_preserves_grad_mode(False)
        assert_query_preserves_grad_mode(True)

    def test_eager_false_is_stable_across_threads_and_grad_modes(self):
        function = torch.jit.is_scripting
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = torch.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    results[index] = (
                        torch.is_grad_enabled(),
                        function(),
                        torch.is_grad_enabled(),
                        function(),
                        torch.is_grad_enabled(),
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
                    False,
                    expected_grad_state,
                    False,
                    expected_grad_state,
                ),
            )
            self.assertIs(result[1], False)
            self.assertIs(result[3], False)

    def test_signature_annotations_documentation_and_internal_ownership(self):
        jit = importlib.import_module("torch_rs.jit")
        internal = importlib.import_module("torch_rs._jit_internal")
        function = jit.is_scripting

        self.assertIs(torch.jit, jit)
        self.assertIs(torch._jit_internal, internal)
        self.assertIs(function, internal.is_scripting)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "() -> bool")
        self.assertEqual(function.__annotations__, {"return": bool})
        self.assertEqual(typing.get_type_hints(function), {"return": bool})
        self.assertEqual(function.__name__, "is_scripting")
        self.assertEqual(function.__qualname__, "is_scripting")
        self.assertEqual(function.__module__, "torch_rs._jit_internal")
        self.assertIs(inspect.getmodule(function), internal)
        self.assertEqual(
            inspect.cleandoc(function.__doc__), inspect.cleandoc(FUNCTION_DOC)
        )
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_exports_copy_and_pickle_use_the_canonical_internal_module(self):
        jit = torch.jit
        function = jit.is_scripting

        self.assertEqual(
            jit.__all__,
            [
                "Attribute",
                "annotate",
                "export",
                "ignore",
                "isinstance",
                "onednn_fusion_enabled",
                "script_if_tracing",
                "strict_fusion",
                "unused",
            ],
        )
        self.assertEqual(
            {name for name in vars(jit) if not name.startswith("_")},
            {
                "Attribute",
                "annotate",
                "export",
                "ignore",
                "isinstance",
                "onednn_fusion_enabled",
                "is_scripting",
                "is_tracing",
                "optimized_execution",
                "script_if_tracing",
                "strict_fusion",
                "unused",
            },
        )

        explicit_namespace = {}
        exec("from torch_rs.jit import is_scripting", explicit_namespace)
        self.assertIs(explicit_namespace["is_scripting"], function)

        wildcard_namespace = {}
        exec("from torch_rs.jit import *", wildcard_namespace)
        self.assertEqual(
            {name for name in wildcard_namespace if not name.startswith("__")},
            {
                "Attribute",
                "annotate",
                "export",
                "ignore",
                "isinstance",
                "onednn_fusion_enabled",
                "script_if_tracing",
                "strict_fusion",
                "unused",
            },
        )
        self.assertNotIn("is_scripting", wildcard_namespace)

        self.assertNotIn("jit", torch.__all__)
        self.assertNotIn("is_scripting", torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("jit", top_level_namespace)
        self.assertNotIn("is_scripting", top_level_namespace)
        self.assertFalse(hasattr(torch, "is_scripting"))

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs._jit_internal", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_rejects_arguments_with_pytorch_2_13_errors(self):
        function = torch.jit.is_scripting
        cases = (
            (
                lambda: function(None),
                "is_scripting() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: function(None, None),
                "is_scripting() takes 0 positional arguments but 2 were given",
            ),
            (
                lambda: function(enabled=True),
                "is_scripting() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: function(None, enabled=True),
                "is_scripting() got an unexpected keyword argument 'enabled'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_scripting_tracing_and_compilation_remain_unsupported(self):
        for name in (
            "CompilationUnit",
            "ScriptFunction",
            "ScriptModule",
            "script",
            "script_method",
            "trace",
            "trace_module",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.jit, name))
        self.assertTrue(callable(torch.compile))

        def exported():
            return "exported"

        def ignored():
            return "ignored"

        def unused():
            return "unused"

        self.assertIs(torch.jit.export(exported), exported)
        self.assertIs(torch.jit.ignore(ignored), ignored)
        self.assertIs(torch.jit.unused(unused), unused)
        self.assertEqual(
            (exported(), ignored(), unused()),
            ("exported", "ignored", "unused"),
        )

    def test_importing_the_package_does_not_import_pytorch(self):
        script = r"""
import sys
import threading

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

assert torch.jit.is_scripting is torch._jit_internal.is_scripting
assert torch.jit.is_scripting() is False
with torch.no_grad():
    assert torch.jit.is_scripting() is False

results = []
thread = threading.Thread(target=lambda: results.append(torch.jit.is_scripting()))
thread.start()
thread.join(timeout=10)
assert not thread.is_alive()
assert results == [False]
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
