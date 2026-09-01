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


FUNCTION_DOC = """Return whether onednn JIT fusion is enabled.

    .. deprecated:: 2.5
        TorchScript is deprecated, please use ``torch.compile`` instead.
    """


class JitOnednnFusionEnabledTests(unittest.TestCase):
    def test_eager_false_is_exact_and_preserves_grad_mode(self):
        function = torch.jit.onednn_fusion_enabled
        self.assertEqual(function.__code__.co_names, ())
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

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
        function = torch.jit.onednn_fusion_enabled
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

    def test_signature_documentation_and_module_identity(self):
        jit = importlib.import_module("torch_rs.jit")
        function = jit.onednn_fusion_enabled

        self.assertIs(torch.jit, jit)
        self.assertIs(sys.modules["torch_rs.jit"], jit)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "()")
        self.assertEqual(function.__annotations__, {})
        self.assertEqual(typing.get_type_hints(function), {})
        self.assertEqual(function.__name__, "onednn_fusion_enabled")
        self.assertEqual(function.__qualname__, "onednn_fusion_enabled")
        self.assertEqual(function.__module__, "torch_rs.jit")
        self.assertIs(inspect.getmodule(function), jit)
        self.assertEqual(
            inspect.cleandoc(function.__doc__), inspect.cleandoc(FUNCTION_DOC)
        )
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_imports_exports_copy_and_pickle_use_the_canonical_jit_module(self):
        jit = torch.jit
        function = jit.onednn_fusion_enabled
        wildcard_supported = {
            "Attribute",
            "annotate",
            "export",
            "ignore",
            "isinstance",
            "onednn_fusion_enabled",
            "script_if_tracing",
            "strict_fusion",
            "unused",
        }

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
                *wildcard_supported,
                "is_scripting",
                "is_tracing",
                "optimized_execution",
            },
        )

        explicit_namespace = {}
        exec(
            "from torch_rs.jit import onednn_fusion_enabled",
            explicit_namespace,
        )
        self.assertIs(explicit_namespace["onednn_fusion_enabled"], function)

        wildcard_namespace = {}
        exec("from torch_rs.jit import *", wildcard_namespace)
        self.assertEqual(
            {
                name
                for name in wildcard_namespace
                if not name.startswith("__")
            },
            wildcard_supported,
        )
        self.assertIs(wildcard_namespace["onednn_fusion_enabled"], function)

        self.assertNotIn("jit", torch.__all__)
        self.assertNotIn("onednn_fusion_enabled", torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("jit", top_level_namespace)
        self.assertNotIn("onednn_fusion_enabled", top_level_namespace)
        self.assertFalse(hasattr(torch, "onednn_fusion_enabled"))

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.jit", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_rejects_arguments_with_pytorch_2_13_errors(self):
        function = torch.jit.onednn_fusion_enabled
        cases = (
            (
                lambda: function(None),
                "onednn_fusion_enabled() takes 0 positional arguments but 1 "
                "was given",
            ),
            (
                lambda: function(None, None),
                "onednn_fusion_enabled() takes 0 positional arguments but 2 "
                "were given",
            ),
            (
                lambda: function(enabled=True),
                "onednn_fusion_enabled() got an unexpected keyword argument "
                "'enabled'",
            ),
            (
                lambda: function(None, enabled=True),
                "onednn_fusion_enabled() got an unexpected keyword argument "
                "'enabled'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_setter_torchscript_and_fusion_execution_remain_unsupported(self):
        self.assertFalse(hasattr(torch.jit, "enable_onednn_fusion"))
        self.assertNotIn("enable_onednn_fusion", torch.jit.__all__)
        for name in (
            "CompilationUnit",
            "ScriptFunction",
            "ScriptModule",
            "freeze",
            "optimize_for_inference",
            "script",
            "script_method",
            "set_fusion_strategy",
            "trace",
            "trace_module",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.jit, name))
        self.assertFalse(hasattr(torch, "compile"))

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

assert torch.jit.onednn_fusion_enabled() is False
assert "onednn_fusion_enabled" in torch.jit.__all__
assert not hasattr(torch.jit, "enable_onednn_fusion")
with torch.no_grad():
    assert torch.jit.onednn_fusion_enabled() is False

results = []
thread = threading.Thread(
    target=lambda: results.append(torch.jit.onednn_fusion_enabled())
)
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
