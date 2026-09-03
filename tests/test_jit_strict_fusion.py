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
import warnings

import torch_rs as torch


STRICT_FUSION_DOC = """Give errors if not all nodes have been fused in inference, or symbolically differentiated in training.

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
    def test_construction_emits_the_eager_warning_at_the_call_site(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warning_line = inspect.currentframe().f_lineno + 1
            context = torch.jit.strict_fusion()

        self.assertIs(type(context), torch.jit.strict_fusion)
        self.assertEqual(context.__dict__, {})
        self.assertEqual(len(caught), 1)
        warning = caught[0]
        self.assertIs(type(warning.message), UserWarning)
        self.assertIs(warning.category, UserWarning)
        self.assertEqual(str(warning.message), "Only works in script mode")
        self.assertEqual(warning.message.args, ("Only works in script mode",))
        self.assertEqual(warning.filename, __file__)
        self.assertEqual(warning.lineno, warning_line)

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            with self.assertRaisesRegex(UserWarning, "^Only works in script mode$"):
                torch.jit.strict_fusion()

    def test_context_is_a_nestable_reusable_no_op_and_propagates_exceptions(self):
        states = [torch.is_grad_enabled()]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            context = torch.jit.strict_fusion()
            with context as first_entered:
                states.append(torch.is_grad_enabled())
                with torch.jit.strict_fusion() as nested_entered:
                    states.append(torch.is_grad_enabled())
                states.append(torch.is_grad_enabled())
            states.append(torch.is_grad_enabled())

            with torch.no_grad():
                states.append(torch.is_grad_enabled())
                with context as second_entered:
                    states.append(torch.is_grad_enabled())
                    with torch.jit.strict_fusion():
                        states.append(torch.is_grad_enabled())
                states.append(torch.is_grad_enabled())
            states.append(torch.is_grad_enabled())

            error = RuntimeError("forwarded failure")
            try:
                with torch.jit.strict_fusion():
                    raise error
            except RuntimeError as raised:
                self.assertIs(raised, error)
            else:
                self.fail("strict_fusion suppressed the context exception")

        self.assertIsNone(first_entered)
        self.assertIsNone(nested_entered)
        self.assertIsNone(second_entered)
        self.assertEqual(
            states,
            [True, True, True, True, True, False, False, False, False, True],
        )

    def test_context_preserves_thread_local_grad_state(self):
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        outcomes = [None] * worker_count
        errors = []

        def worker(index):
            try:
                grad_context = (
                    torch.no_grad() if index % 2 else contextlib.nullcontext()
                )
                with grad_context:
                    barrier.wait(timeout=10)
                    before = torch.is_grad_enabled()
                    with torch.jit.strict_fusion() as entered:
                        inside = torch.is_grad_enabled()
                    outcomes[index] = (
                        before,
                        inside,
                        torch.is_grad_enabled(),
                        entered,
                    )
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))

        threads = [
            threading.Thread(target=worker, args=(index,))
            for index in range(worker_count)
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(
            outcomes,
            [
                (True, True, True, None),
                (False, False, False, None),
                (True, True, True, None),
                (False, False, False, None),
                (True, True, True, None),
                (False, False, False, None),
                (True, True, True, None),
                (False, False, False, None),
            ],
        )
        self.assertIs(torch.is_grad_enabled(), True)

    def test_signature_documentation_and_module_ownership(self):
        jit = importlib.import_module("torch_rs.jit")
        strict_fusion = jit.strict_fusion

        self.assertIs(torch.jit, jit)
        self.assertIs(sys.modules["torch_rs.jit"], jit)
        self.assertIs(type(strict_fusion), type)
        self.assertEqual(strict_fusion.__bases__, (object,))
        self.assertEqual(strict_fusion.__name__, "strict_fusion")
        self.assertEqual(strict_fusion.__qualname__, "strict_fusion")
        self.assertEqual(strict_fusion.__module__, "torch_rs.jit")
        self.assertIs(inspect.getmodule(strict_fusion), jit)
        self.assertEqual(str(inspect.signature(strict_fusion)), "() -> None")
        self.assertEqual(strict_fusion.__annotations__, {})
        self.assertEqual(typing.get_type_hints(strict_fusion), {})
        self.assertEqual(
            inspect.cleandoc(strict_fusion.__doc__), STRICT_FUSION_DOC.strip()
        )
        self.assertIsNone(strict_fusion.__text_signature__)
        for name in (
            "__module__",
            "__doc__",
            "__init__",
            "__enter__",
            "__exit__",
            "__dict__",
            "__weakref__",
        ):
            with self.subTest(class_field=name):
                self.assertIn(name, strict_fusion.__dict__)
        self.assertEqual(
            {
                name
                for name in strict_fusion.__dict__
                if not name.startswith("__")
            },
            set(),
        )

        methods = (
            (
                strict_fusion.__init__,
                "strict_fusion.__init__",
                "(self) -> None",
                {"return": None},
                {"return": type(None)},
            ),
            (
                strict_fusion.__enter__,
                "strict_fusion.__enter__",
                "(self)",
                {},
                {},
            ),
            (
                strict_fusion.__exit__,
                "strict_fusion.__exit__",
                "(self, type: Any, value: Any, tb: Any) -> None",
                {
                    "type": typing.Any,
                    "value": typing.Any,
                    "tb": typing.Any,
                    "return": None,
                },
                {
                    "type": typing.Any,
                    "value": typing.Any,
                    "tb": typing.Any,
                    "return": type(None),
                },
            ),
        )
        for method, qualname, signature, annotations, type_hints in methods:
            with self.subTest(method=method.__name__):
                self.assertIs(type(method), types.FunctionType)
                self.assertEqual(method.__qualname__, qualname)
                self.assertEqual(method.__module__, "torch_rs.jit")
                self.assertIs(inspect.getmodule(method), jit)
                self.assertEqual(str(inspect.signature(method)), signature)
                self.assertEqual(method.__annotations__, annotations)
                self.assertEqual(typing.get_type_hints(method), type_hints)
                self.assertIsNone(method.__doc__)
                self.assertIsNone(method.__defaults__)
                self.assertIsNone(method.__kwdefaults__)
                self.assertEqual(method.__dict__, {})

    def test_exports_copying_and_pickling_use_the_canonical_class(self):
        jit = torch.jit
        strict_fusion = jit.strict_fusion
        supported = {
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
                *supported,
                "is_scripting",
                "is_tracing",
                "optimized_execution",
            },
        )
        namespace = {}
        exec("from torch_rs.jit import *", namespace)
        self.assertEqual(
            {name for name in namespace if not name.startswith("__")}, supported
        )
        self.assertIs(namespace["strict_fusion"], strict_fusion)

        self.assertNotIn("strict_fusion", torch.__all__)
        self.assertFalse(hasattr(torch, "strict_fusion"))
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("strict_fusion", top_level_namespace)

        self.assertIs(copy.copy(strict_fusion), strict_fusion)
        self.assertIs(copy.deepcopy(strict_fusion), strict_fusion)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(kind="class", protocol=protocol):
                payload = pickle.dumps(strict_fusion, protocol=protocol)
                self.assertIn(b"torch_rs.jit", payload)
                self.assertIs(pickle.loads(payload), strict_fusion)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            context = strict_fusion()
        value = {"items": [1, 2]}
        context.value = value
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            shallow = copy.copy(context)
            deep = copy.deepcopy(context)
        self.assertEqual(caught, [])
        self.assertIsNot(shallow, context)
        self.assertIs(type(shallow), strict_fusion)
        self.assertIs(shallow.value, value)
        self.assertIsNot(deep, context)
        self.assertIs(type(deep), strict_fusion)
        self.assertEqual(deep.value, value)
        self.assertIsNot(deep.value, value)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(kind="instance", protocol=protocol):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    payload = pickle.dumps(context, protocol=protocol)
                    restored = pickle.loads(payload)
                self.assertEqual(caught, [])
                self.assertIn(b"torch_rs.jit", payload)
                self.assertIs(type(restored), strict_fusion)
                self.assertEqual(restored.value, value)
                self.assertIsNot(restored.value, value)

    def test_invalid_constructor_calls_match_pytorch_2_13_errors(self):
        strict_fusion = torch.jit.strict_fusion
        cases = (
            (
                lambda: strict_fusion(1),
                "strict_fusion.__init__() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: strict_fusion(1, 2),
                "strict_fusion.__init__() takes 1 positional argument but 3 were given",
            ),
            (
                lambda: strict_fusion(value=1),
                "strict_fusion.__init__() got an unexpected keyword argument 'value'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_internal_scripting_predicate_controls_only_the_eager_warning(self):
        jit = torch.jit
        internal = torch._jit_internal
        original_internal = internal.is_scripting
        original_public = jit.is_scripting
        try:
            internal.is_scripting = lambda: True
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                context = jit.strict_fusion()
            self.assertEqual(caught, [])
            self.assertIs(type(context), jit.strict_fusion)

            internal.is_scripting = original_internal
            jit.is_scripting = lambda: True
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                jit.strict_fusion()
            self.assertEqual(len(caught), 1)
            self.assertEqual(str(caught[0].message), "Only works in script mode")
        finally:
            internal.is_scripting = original_internal
            jit.is_scripting = original_public

    def test_scripting_tracing_and_fusion_execution_remain_unsupported(self):
        self.assertTrue(callable(torch.jit.strict_fusion))
        self.assertIs(torch.jit.is_scripting(), False)
        self.assertIs(torch.jit.is_tracing(), False)
        for name in (
            "CompilationUnit",
            "ScriptFunction",
            "ScriptModule",
            "enable_onednn_fusion",
            "script",
            "set_fusion_strategy",
            "trace",
            "trace_module",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.jit, name))
        self.assertTrue(callable(torch.compile))

    def test_importing_the_package_does_not_import_pytorch(self):
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
    with torch.jit.strict_fusion() as entered:
        assert entered is None
assert len(caught) == 1
assert type(caught[0].message) is UserWarning
assert str(caught[0].message) == "Only works in script mode"
assert torch.jit.strict_fusion.__module__ == "torch_rs.jit"
assert not hasattr(torch.jit, "script")
assert not hasattr(torch.jit, "set_fusion_strategy")
assert callable(torch.compile)
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
