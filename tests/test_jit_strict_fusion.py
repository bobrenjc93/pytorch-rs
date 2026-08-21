import contextlib
import copy
import importlib
import inspect
import os
import pickle
import subprocess
import sys
import threading
import types
import typing
import unittest
import warnings
from unittest import mock

import torch_rs as torch


STRICT_FUSION_DOC = """
    Give errors if not all nodes have been fused in inference, or symbolically differentiated in training.

    .. deprecated:: 2.5
        TorchScript is deprecated, please use ``torch.compile`` instead.

    Example:
    Forcing fusion of additions.

    .. code-block:: python

        @torch.jit.script
        def foo(x):
            with torch.jit.strict_fusion():
                return x + x + x

    """


class JitStrictFusionTests(unittest.TestCase):
    def make_context(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return torch.jit.strict_fusion()

    def test_constructor_emits_the_exact_eager_warning(self):
        before = torch.is_grad_enabled()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warning_line = inspect.currentframe().f_lineno + 1
            context = torch.jit.strict_fusion()

        self.assertIsInstance(context, torch.jit.strict_fusion)
        self.assertIs(torch.is_grad_enabled(), before)
        self.assertEqual(len(caught), 1)
        warning = caught[0]
        self.assertIs(warning.category, UserWarning)
        self.assertEqual(str(warning.message), "Only works in script mode")
        self.assertEqual(warning.message.args, ("Only works in script mode",))
        self.assertEqual(os.path.realpath(warning.filename), os.path.realpath(__file__))
        self.assertEqual(warning.lineno, warning_line)

    def test_context_is_reentrant_nested_and_preserves_grad_mode(self):
        context = self.make_context()

        def assert_noop(expected_grad_state):
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)
            with context as entered:
                self.assertIsNone(entered)
                self.assertIs(torch.is_grad_enabled(), expected_grad_state)
                with context as nested_entered:
                    self.assertIsNone(nested_entered)
                    self.assertIs(torch.is_grad_enabled(), expected_grad_state)
                self.assertIs(torch.is_grad_enabled(), expected_grad_state)
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)

        assert_noop(True)
        with torch.no_grad():
            assert_noop(False)
        assert_noop(True)

        self.assertIsNone(context.__enter__())
        self.assertIsNone(context.__exit__(None, None, None))

    def test_shared_context_preserves_thread_local_grad_states(self):
        context = self.make_context()
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                mode = torch.no_grad() if index % 2 else contextlib.nullcontext()
                initial = torch.is_grad_enabled()
                with mode:
                    expected = index % 2 == 0
                    before = torch.is_grad_enabled()
                    barrier.wait(timeout=10)
                    with context as entered:
                        during = torch.is_grad_enabled()
                    after = torch.is_grad_enabled()
                final = torch.is_grad_enabled()
                results[index] = (initial, before, entered, during, after, final)
                self.assertIs(before, expected)
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
            expected = index % 2 == 0
            self.assertEqual(result, (True, expected, None, expected, expected, True))

    def test_exceptions_propagate_without_changing_grad_mode(self):
        context = self.make_context()
        marker = RuntimeError("strict fusion body failed")

        with self.assertRaises(RuntimeError) as raised:
            with context:
                self.assertIs(torch.is_grad_enabled(), True)
                raise marker
        self.assertIs(raised.exception, marker)
        self.assertIs(torch.is_grad_enabled(), True)

        with torch.no_grad():
            with self.assertRaises(RuntimeError) as raised:
                with context:
                    self.assertIs(torch.is_grad_enabled(), False)
                    raise marker
            self.assertIs(raised.exception, marker)
            self.assertIs(torch.is_grad_enabled(), False)
        self.assertIs(torch.is_grad_enabled(), True)

    def test_signature_documentation_and_ownership(self):
        jit = importlib.import_module("torch_rs.jit")
        strict_fusion = jit.strict_fusion

        self.assertIs(torch.jit, jit)
        self.assertIs(type(strict_fusion), type)
        self.assertEqual(strict_fusion.__bases__, (object,))
        self.assertEqual(str(inspect.signature(strict_fusion)), "() -> None")
        self.assertEqual(str(inspect.signature(strict_fusion.__init__)), "(self) -> None")
        self.assertEqual(str(inspect.signature(strict_fusion.__enter__)), "(self)")
        self.assertEqual(
            str(inspect.signature(strict_fusion.__exit__)),
            "(self, type: Any, value: Any, tb: Any) -> None",
        )
        self.assertEqual(strict_fusion.__annotations__, {})
        self.assertEqual(strict_fusion.__init__.__annotations__, {"return": None})
        self.assertEqual(strict_fusion.__enter__.__annotations__, {})
        self.assertEqual(
            strict_fusion.__exit__.__annotations__,
            {
                "type": typing.Any,
                "value": typing.Any,
                "tb": typing.Any,
                "return": None,
            },
        )
        self.assertEqual(
            typing.get_type_hints(strict_fusion.__exit__),
            {
                "type": typing.Any,
                "value": typing.Any,
                "tb": typing.Any,
                "return": type(None),
            },
        )
        self.assertEqual(strict_fusion.__name__, "strict_fusion")
        self.assertEqual(strict_fusion.__qualname__, "strict_fusion")
        self.assertEqual(strict_fusion.__module__, "torch_rs.jit")
        self.assertIs(inspect.getmodule(strict_fusion), jit)
        self.assertEqual(
            inspect.cleandoc(strict_fusion.__doc__),
            inspect.cleandoc(STRICT_FUSION_DOC),
        )

        for name in ("__init__", "__enter__", "__exit__"):
            with self.subTest(name=name):
                method = getattr(strict_fusion, name)
                self.assertIs(type(method), types.FunctionType)
                self.assertEqual(method.__module__, "torch_rs.jit")
                self.assertEqual(method.__qualname__, f"strict_fusion.{name}")
                self.assertIs(inspect.getmodule(method), jit)
                self.assertIsNone(method.__doc__)
                self.assertIsNone(method.__defaults__)
                self.assertIsNone(method.__kwdefaults__)
                self.assertEqual(method.__dict__, {})

    def test_exports_match_the_supported_jit_namespace(self):
        jit = torch.jit
        strict_fusion = jit.strict_fusion
        wildcard_supported = {
            "Attribute",
            "annotate",
            "export",
            "ignore",
            "isinstance",
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
                "script_if_tracing",
                "strict_fusion",
                "unused",
            ],
        )
        self.assertEqual(
            {name for name in vars(jit) if not name.startswith("_")},
            {*wildcard_supported, "is_scripting", "is_tracing"},
        )

        direct_namespace = {}
        exec("from torch_rs.jit import strict_fusion", direct_namespace)
        self.assertIs(direct_namespace["strict_fusion"], strict_fusion)

        wildcard_namespace = {}
        exec("from torch_rs.jit import *", wildcard_namespace)
        self.assertEqual(
            {name for name in wildcard_namespace if not name.startswith("__")},
            wildcard_supported,
        )
        self.assertIs(wildcard_namespace["strict_fusion"], strict_fusion)

        self.assertNotIn("strict_fusion", torch.__all__)
        self.assertFalse(hasattr(torch, "strict_fusion"))
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("strict_fusion", top_level_namespace)

    def test_copying_and_pickling_preserve_plain_context_state(self):
        strict_fusion = torch.jit.strict_fusion
        context = self.make_context()
        context.payload = {"items": [1, 2]}

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            shallow = copy.copy(context)
            deep = copy.deepcopy(context)

        self.assertEqual(caught, [])
        self.assertIsNot(shallow, context)
        self.assertIs(type(shallow), strict_fusion)
        self.assertIs(shallow.payload, context.payload)
        self.assertIsNot(deep, context)
        self.assertIs(type(deep), strict_fusion)
        self.assertEqual(deep.payload, context.payload)
        self.assertIsNot(deep.payload, context.payload)
        self.assertIsNot(deep.payload["items"], context.payload["items"])

        self.assertIs(copy.copy(strict_fusion), strict_fusion)
        self.assertIs(copy.deepcopy(strict_fusion), strict_fusion)
        for method_name in ("__init__", "__enter__", "__exit__"):
            method = getattr(strict_fusion, method_name)
            self.assertIs(copy.copy(method), method)
            self.assertIs(copy.deepcopy(method), method)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                class_payload = pickle.dumps(strict_fusion, protocol=protocol)
                self.assertIn(b"torch_rs.jit", class_payload)
                self.assertIs(pickle.loads(class_payload), strict_fusion)

                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    payload = pickle.dumps(context, protocol=protocol)
                    restored = pickle.loads(payload)
                self.assertEqual(caught, [])
                self.assertIn(b"torch_rs.jit", payload)
                self.assertIsNot(restored, context)
                self.assertIs(type(restored), strict_fusion)
                self.assertEqual(restored.__dict__, context.__dict__)
                self.assertIsNot(restored.payload, context.payload)
                with restored as entered:
                    self.assertIsNone(entered)

    def test_argument_errors_match_pytorch_2_13(self):
        strict_fusion = torch.jit.strict_fusion
        cases = (
            (
                lambda: strict_fusion(None),
                "strict_fusion.__init__() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: strict_fusion(None, None),
                "strict_fusion.__init__() takes 1 positional argument but 3 were given",
            ),
            (
                lambda: strict_fusion(enabled=True),
                "strict_fusion.__init__() got an unexpected keyword argument 'enabled'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    with self.assertRaises(TypeError) as raised:
                        call()
                self.assertEqual(caught, [])
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_warning_is_suppressed_only_for_the_scripting_state(self):
        with mock.patch.object(
            torch._jit_internal, "is_scripting", return_value=True
        ) as is_scripting:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                context = torch.jit.strict_fusion()
        is_scripting.assert_called_once_with()
        self.assertEqual(caught, [])
        with context as entered:
            self.assertIsNone(entered)

    def test_importing_and_using_the_context_does_not_import_pytorch(self):
        script = r"""
import sys
import warnings

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    context = torch.jit.strict_fusion()
assert len(caught) == 1
assert caught[0].category is UserWarning
assert str(caught[0].message) == "Only works in script mode"
before = torch.is_grad_enabled()
with context as entered:
    assert entered is None
    assert torch.is_grad_enabled() is before
assert torch.is_grad_enabled() is before
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
