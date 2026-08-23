import copy
import importlib
import inspect
import math
import pickle
import subprocess
import sys
import threading
import types
import typing
import unittest

import torch_rs as torch


SETTER_DOC = """Enable or disables onednn JIT fusion based on the parameter `enabled`.

    .. deprecated:: 2.5
        TorchScript is deprecated, please use ``torch.compile`` instead.
    """


GETTER_DOC = """Return whether onednn JIT fusion is enabled.

    .. deprecated:: 2.5
        TorchScript is deprecated, please use ``torch.compile`` instead.
    """


class _BoolValue:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def __bool__(self):
        self.calls += 1
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result

    def __repr__(self):
        return f"_BoolValue({self.result!r})"


class _LengthOnlyValue:
    def __init__(self, length):
        self.length = length
        self.calls = 0

    def __len__(self):
        self.calls += 1
        return self.length

    def __repr__(self):
        return f"_LengthOnlyValue({self.length!r})"


class _OpaqueValue:
    def __repr__(self):
        return "_OpaqueValue()"


class _RaisingReprValue:
    def __repr__(self):
        raise RuntimeError("repr failed")


class JitOnednnFusionEnabledTests(unittest.TestCase):
    def setUp(self):
        self.original = torch.jit.onednn_fusion_enabled()
        torch.jit.enable_onednn_fusion(False)

    def tearDown(self):
        torch.jit.enable_onednn_fusion(self.original)

    def test_mutation_returns_none_and_preserves_grad_mode(self):
        setter = torch.jit.enable_onednn_fusion
        getter = torch.jit.onednn_fusion_enabled

        def assert_update(enabled, expected, expected_grad_state):
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)
            self.assertIs(setter(enabled), None)
            self.assertIs(getter(), expected)
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)

        assert_update(True, True, True)
        with torch.no_grad():
            assert_update(False, False, False)
            assert_update(1, True, False)
            assert_update(None, False, False)
        assert_update(-1.5, True, True)
        assert_update(0j, False, True)

    def test_pytorch_bool_conversion_and_non_mutating_errors(self):
        setter = torch.jit.enable_onednn_fusion
        getter = torch.jit.onednn_fusion_enabled
        true_value = _BoolValue(True)
        false_value = _BoolValue(False)
        accepted = (
            (False, False),
            (True, True),
            (0, False),
            (2, True),
            (0.0, False),
            (math.nan, True),
            (0j, False),
            (1j, True),
            (None, False),
            (true_value, True),
            (false_value, False),
            (torch.tensor([0.0]), False),
            (torch.tensor([2.0]), True),
        )
        for value, expected in accepted:
            with self.subTest(value=repr(value)):
                self.assertIs(setter(value), None)
                self.assertIs(getter(), expected)

        self.assertEqual(true_value.calls, 1)
        self.assertEqual(false_value.calls, 1)

        length_only = _LengthOnlyValue(1)
        invalid_bool = _BoolValue(1)
        raising_bool = _BoolValue(ValueError("bool failed"))
        invalid = (
            "",
            "enabled",
            [],
            _OpaqueValue(),
            length_only,
            invalid_bool,
            raising_bool,
            _RaisingReprValue(),
            torch.tensor([1.0, 2.0]),
        )
        for marker in (False, True):
            for value in invalid:
                with self.subTest(marker=marker, value=type(value).__name__):
                    setter(marker)
                    try:
                        value_repr = repr(value)
                    except BaseException:
                        value_repr = "<repr raised Error>"
                    message = (
                        "_jit_set_llga_enabled(): incompatible function arguments. "
                        "The following argument types are supported:\n"
                        "    1. (arg0: bool) -> bool\n\n"
                        f"Invoked with: {value_repr}"
                    )
                    with self.assertRaises(TypeError) as raised:
                        setter(value)
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))
                    self.assertIsNone(raised.exception.__context__)
                    self.assertIsNone(raised.exception.__cause__)
                    self.assertIs(getter(), marker)

        self.assertEqual(length_only.calls, 0)
        self.assertEqual(invalid_bool.calls, 2)
        self.assertEqual(raising_bool.calls, 2)

    def test_updates_are_process_global_across_threads(self):
        jit = torch.jit
        worker_ready = threading.Event()
        read_updated = threading.Event()
        observations = []
        errors = []

        def observer():
            try:
                observations.append(jit.onednn_fusion_enabled())
                worker_ready.set()
                if not read_updated.wait(timeout=10):
                    raise RuntimeError("timed out waiting for oneDNN state update")
                observations.append(jit.onednn_fusion_enabled())
                observations.append(jit.enable_onednn_fusion(False))
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=observer)
        thread.start()
        self.assertTrue(worker_ready.wait(timeout=10))
        self.assertIs(jit.enable_onednn_fusion(True), None)
        read_updated.set()
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(observations, [False, True, None])
        self.assertIs(jit.onednn_fusion_enabled(), False)

    def test_reload_and_reimport_share_state_with_old_functions(self):
        original_module = torch.jit
        old_getter = original_module.onednn_fusion_enabled
        old_setter = original_module.enable_onednn_fusion

        old_setter(True)
        self.assertIs(importlib.reload(original_module), original_module)
        self.assertIs(torch.jit, original_module)
        self.assertIs(old_getter(), True)
        self.assertIs(original_module.onednn_fusion_enabled(), True)

        self.assertIs(original_module.enable_onednn_fusion(False), None)
        self.assertIs(old_getter(), False)

        module_name = original_module.__name__
        try:
            self.assertIs(sys.modules.pop(module_name), original_module)
            replacement_module = importlib.import_module(module_name)
            self.assertIsNot(replacement_module, original_module)
            self.assertIs(torch.jit, replacement_module)
            self.assertIs(replacement_module.onednn_fusion_enabled(), False)
            self.assertIs(old_getter(), False)

            self.assertIs(replacement_module.enable_onednn_fusion(True), None)
            self.assertIs(old_getter(), True)
            self.assertIs(original_module.onednn_fusion_enabled(), True)
            self.assertIs(old_setter(False), None)
            self.assertIs(replacement_module.onednn_fusion_enabled(), False)
        finally:
            sys.modules[module_name] = original_module
            torch.jit = original_module

    def test_eager_results_and_gradients_are_unchanged(self):
        def eager_outcome(enabled):
            self.assertIs(torch.jit.enable_onednn_fusion(enabled), None)
            leaf = torch.tensor([-2.0, 0.0, 3.0], requires_grad=True)
            output = leaf.square()
            output.sum().backward()
            return output.tolist(), leaf.grad.tolist(), torch.is_grad_enabled()

        self.assertEqual(eager_outcome(False), eager_outcome(True))

    def test_signature_documentation_and_module_identity(self):
        jit = importlib.import_module("torch_rs.jit")
        setter = jit.enable_onednn_fusion
        getter = jit.onednn_fusion_enabled

        self.assertIs(torch.jit, jit)
        self.assertIs(sys.modules["torch_rs.jit"], jit)
        for function in (setter, getter):
            self.assertIs(type(function), types.FunctionType)
            self.assertEqual(function.__module__, "torch_rs.jit")
            self.assertIs(inspect.getmodule(function), jit)
            self.assertIsNone(function.__defaults__)
            self.assertIsNone(function.__kwdefaults__)
            self.assertEqual(function.__dict__, {})
            self.assertFalse(hasattr(function, "__text_signature__"))
            self.assertEqual(function.__code__.co_freevars, ())
            self.assertEqual(function.__code__.co_cellvars, ())

        self.assertEqual(str(inspect.signature(setter)), "(enabled: bool) -> None")
        self.assertEqual(setter.__annotations__, {"enabled": bool, "return": None})
        self.assertEqual(
            typing.get_type_hints(setter),
            {"enabled": bool, "return": type(None)},
        )
        self.assertEqual(setter.__name__, "enable_onednn_fusion")
        self.assertEqual(setter.__qualname__, "enable_onednn_fusion")
        self.assertEqual(
            inspect.cleandoc(setter.__doc__), inspect.cleandoc(SETTER_DOC)
        )

        self.assertEqual(str(inspect.signature(getter)), "()")
        self.assertEqual(getter.__annotations__, {})
        self.assertEqual(typing.get_type_hints(getter), {})
        self.assertEqual(getter.__name__, "onednn_fusion_enabled")
        self.assertEqual(getter.__qualname__, "onednn_fusion_enabled")
        self.assertEqual(
            inspect.cleandoc(getter.__doc__), inspect.cleandoc(GETTER_DOC)
        )

    def test_imports_exports_copy_and_pickle_use_the_canonical_jit_module(self):
        jit = torch.jit
        wildcard_supported = {
            "Attribute",
            "annotate",
            "enable_onednn_fusion",
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
                "enable_onednn_fusion",
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
            {*wildcard_supported, "is_scripting", "is_tracing"},
        )

        explicit_namespace = {}
        exec(
            "from torch_rs.jit import enable_onednn_fusion, onednn_fusion_enabled",
            explicit_namespace,
        )
        self.assertIs(
            explicit_namespace["enable_onednn_fusion"], jit.enable_onednn_fusion
        )
        self.assertIs(
            explicit_namespace["onednn_fusion_enabled"], jit.onednn_fusion_enabled
        )

        wildcard_namespace = {}
        exec("from torch_rs.jit import *", wildcard_namespace)
        self.assertEqual(
            {name for name in wildcard_namespace if not name.startswith("__")},
            wildcard_supported,
        )
        for name in wildcard_supported:
            self.assertIs(wildcard_namespace[name], getattr(jit, name))

        self.assertNotIn("jit", torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        for name in ("enable_onednn_fusion", "onednn_fusion_enabled"):
            self.assertNotIn(name, torch.__all__)
            self.assertNotIn(name, top_level_namespace)
            self.assertFalse(hasattr(torch, name))

        for function in (jit.enable_onednn_fusion, jit.onednn_fusion_enabled):
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(function=function.__name__, protocol=protocol):
                    payload = pickle.dumps(function, protocol=protocol)
                    self.assertIn(b"torch_rs.jit", payload)
                    self.assertIs(pickle.loads(payload), function)

    def test_call_shape_errors_do_not_mutate_state(self):
        setter = torch.jit.enable_onednn_fusion
        getter = torch.jit.onednn_fusion_enabled
        setter(True)
        cases = (
            (
                lambda: setter(),
                "enable_onednn_fusion() missing 1 required positional argument: "
                "'enabled'",
            ),
            (
                lambda: setter(True, False),
                "enable_onednn_fusion() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: setter(value=True),
                "enable_onednn_fusion() got an unexpected keyword argument 'value'",
            ),
            (
                lambda: setter(True, enabled=False),
                "enable_onednn_fusion() got multiple values for argument 'enabled'",
            ),
            (
                lambda: getter(None),
                "onednn_fusion_enabled() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: getter(enabled=True),
                "onednn_fusion_enabled() got an unexpected keyword argument 'enabled'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertIs(getter(), True)

    def test_torchscript_and_fusion_execution_remain_unsupported(self):
        self.assertTrue(callable(torch.jit.enable_onednn_fusion))
        self.assertTrue(callable(torch.jit.onednn_fusion_enabled))
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

    def test_fresh_process_defaults_false_without_importing_pytorch(self):
        script = r'''
import importlib
import sys
import threading

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

jit = torch.jit
old_getter = jit.onednn_fusion_enabled
assert old_getter() is False
assert jit.enable_onednn_fusion(True) is None
assert old_getter() is True
assert importlib.reload(jit) is jit
assert jit.onednn_fusion_enabled() is True
assert old_getter() is True

results = []
thread = threading.Thread(
    target=lambda: results.extend(
        [jit.enable_onednn_fusion(None), jit.onednn_fusion_enabled()]
    )
)
thread.start()
thread.join(timeout=10)
assert not thread.is_alive()
assert results == [None, False]
assert old_getter() is False
assert "enable_onednn_fusion" in jit.__all__
assert "onednn_fusion_enabled" in jit.__all__
assert not hasattr(torch, "enable_onednn_fusion")
assert not hasattr(torch, "onednn_fusion_enabled")
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
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
