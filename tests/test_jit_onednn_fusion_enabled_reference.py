import copy
import importlib
import inspect
import math
import pickle
import pickletools
import sys
import threading
import types
import typing
import unittest

import numpy as np

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


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


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class JitOnednnFusionEnabledReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "jit oneDNN fusion differentials require pinned PyTorch 2.13.0"
            )

    def setUp(self):
        self.original_actual = torch.jit.onednn_fusion_enabled()
        self.original_expected = reference_torch.jit.onednn_fusion_enabled()
        torch.jit.enable_onednn_fusion(False)
        reference_torch.jit.enable_onednn_fusion(False)

    def tearDown(self):
        torch.jit.enable_onednn_fusion(self.original_actual)
        reference_torch.jit.enable_onednn_fusion(self.original_expected)

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(BaseException) as actual_raised:
            actual_call()
        with self.assertRaises(BaseException) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)
        self.assertEqual(
            actual_raised.exception.__context__, expected_raised.exception.__context__
        )
        self.assertEqual(
            actual_raised.exception.__cause__, expected_raised.exception.__cause__
        )

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

    def eager_outcome(self, module, enabled):
        before = module.is_grad_enabled()
        set_result = module.jit.enable_onednn_fusion(enabled)
        state = module.jit.onednn_fusion_enabled()
        leaf = module.tensor([-2.0, 0.0, 3.0], requires_grad=True)
        output = leaf.square()
        output.sum().backward()
        return (
            before,
            set_result,
            state,
            type(state) is bool,
            output.tolist(),
            leaf.grad.tolist(),
            module.is_grad_enabled(),
        )

    def conversion_outcome(self, module, value, marker):
        module.jit.enable_onednn_fusion(marker)
        try:
            result = module.jit.enable_onednn_fusion(value)
        except BaseException as error:
            outcome = (
                "error",
                type(error).__name__,
                str(error),
                error.args,
                error.__context__,
                error.__cause__,
            )
        else:
            outcome = ("ok", result)
        return (
            outcome,
            module.jit.onednn_fusion_enabled(),
            getattr(value, "calls", None),
        )

    def threaded_outcome(self, module):
        module.jit.enable_onednn_fusion(False)
        worker_ready = threading.Event()
        read_updated = threading.Event()
        observations = []
        errors = []

        def observer():
            try:
                observations.append(module.jit.onednn_fusion_enabled())
                worker_ready.set()
                if not read_updated.wait(timeout=10):
                    raise RuntimeError("timed out waiting for oneDNN state update")
                observations.append(module.jit.onednn_fusion_enabled())
                observations.append(module.jit.enable_onednn_fusion(False))
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))

        thread = threading.Thread(target=observer)
        thread.start()
        self.assertTrue(worker_ready.wait(timeout=10))
        set_result = module.jit.enable_onednn_fusion(True)
        read_updated.set()
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        return set_result, observations, module.jit.onednn_fusion_enabled()

    def test_state_mutation_eager_results_gradients_and_threads_match(self):
        for enabled in (False, True, 0, 3, 0.0, math.nan, 0j, 1j, None):
            with self.subTest(enabled=enabled):
                self.assertEqual(
                    self.eager_outcome(torch, enabled),
                    self.eager_outcome(reference_torch, enabled),
                )

        self.assertEqual(
            self.threaded_outcome(torch),
            self.threaded_outcome(reference_torch),
        )

    def test_bool_conversion_and_non_mutating_errors_match(self):
        accepted_factories = (
            lambda module: np.bool_(True),
            lambda module: np.int64(0),
            lambda module: np.int64(2),
            lambda module: np.float64(0.0),
            lambda module: _BoolValue(False),
            lambda module: _BoolValue(True),
            lambda module: module.tensor([0.0]),
            lambda module: module.tensor([2.0]),
        )
        invalid_factories = (
            lambda module: "",
            lambda module: "enabled",
            lambda module: [],
            lambda module: _OpaqueValue(),
            lambda module: _LengthOnlyValue(1),
            lambda module: _BoolValue(1),
            lambda module: _BoolValue(ValueError("bool failed")),
            lambda module: _RaisingReprValue(),
        )

        for marker in (False, True):
            for case, factory in enumerate((*accepted_factories, *invalid_factories)):
                with self.subTest(marker=marker, case=case):
                    actual_value = factory(torch)
                    expected_value = factory(reference_torch)
                    self.assertEqual(
                        self.conversion_outcome(torch, actual_value, marker),
                        self.conversion_outcome(
                            reference_torch, expected_value, marker
                        ),
                    )

    def test_reload_and_reimport_semantics_match(self):
        for package in (torch, reference_torch):
            with self.subTest(package=package.__name__):
                original_module = package.jit
                old_getter = original_module.onednn_fusion_enabled
                old_setter = original_module.enable_onednn_fusion
                old_setter(True)

                try:
                    self.assertIs(importlib.reload(original_module), original_module)
                    self.assertIs(package.jit, original_module)
                    self.assertIs(old_getter(), True)
                    self.assertIs(original_module.onednn_fusion_enabled(), True)

                    self.assertIs(
                        original_module.enable_onednn_fusion(False), None
                    )
                    self.assertIs(old_getter(), False)

                    module_name = original_module.__name__
                    self.assertIs(sys.modules.pop(module_name), original_module)
                    replacement_module = importlib.import_module(module_name)
                    self.assertIsNot(replacement_module, original_module)
                    self.assertIs(package.jit, replacement_module)
                    self.assertIs(
                        replacement_module.onednn_fusion_enabled(), False
                    )
                    self.assertIs(replacement_module.enable_onednn_fusion(True), None)
                    self.assertIs(old_getter(), True)
                    self.assertIs(old_setter(False), None)
                    self.assertIs(
                        replacement_module.onednn_fusion_enabled(), False
                    )
                finally:
                    sys.modules[original_module.__name__] = original_module
                    package.jit = original_module

    def test_signature_documentation_and_identity_match(self):
        actual_jit = importlib.import_module("torch_rs.jit")
        expected_jit = importlib.import_module("torch.jit")

        for name in ("enable_onednn_fusion", "onednn_fusion_enabled"):
            with self.subTest(name=name):
                actual = getattr(actual_jit, name)
                expected = getattr(expected_jit, name)
                self.assertIs(type(actual), types.FunctionType)
                self.assertIs(type(expected), types.FunctionType)
                self.assertEqual(
                    str(inspect.signature(actual)), str(inspect.signature(expected))
                )
                self.assertEqual(actual.__annotations__, expected.__annotations__)
                self.assertEqual(
                    typing.get_type_hints(actual), typing.get_type_hints(expected)
                )
                self.assertEqual(actual.__name__, expected.__name__)
                self.assertEqual(actual.__qualname__, expected.__qualname__)
                self.assertEqual(
                    actual.__module__.replace("torch_rs", "torch"),
                    expected.__module__,
                )
                self.assertIs(inspect.getmodule(actual), actual_jit)
                self.assertIs(inspect.getmodule(expected), expected_jit)
                self.assertEqual(actual.__doc__, expected.__doc__)
                self.assertEqual(actual.__defaults__, expected.__defaults__)
                self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
                self.assertEqual(actual.__dict__, expected.__dict__)
                self.assertEqual(
                    hasattr(actual, "__text_signature__"),
                    hasattr(expected, "__text_signature__"),
                )

    def test_imports_exports_copy_and_pickle_match(self):
        actual_jit = torch.jit
        expected_jit = reference_torch.jit
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
        public_supported = {*wildcard_supported, "is_scripting", "is_tracing"}

        self.assertEqual(
            actual_jit.__all__,
            [name for name in expected_jit.__all__ if name in wildcard_supported],
        )
        self.assertEqual(
            {name for name in vars(actual_jit) if not name.startswith("_")},
            public_supported,
        )

        actual_explicit = {}
        expected_explicit = {}
        exec(
            "from torch_rs.jit import enable_onednn_fusion, onednn_fusion_enabled",
            actual_explicit,
        )
        exec(
            "from torch.jit import enable_onednn_fusion, onednn_fusion_enabled",
            expected_explicit,
        )
        for name in ("enable_onednn_fusion", "onednn_fusion_enabled"):
            self.assertIs(actual_explicit[name], getattr(actual_jit, name))
            self.assertIs(expected_explicit[name], getattr(expected_jit, name))

        actual_namespace = {}
        expected_namespace = {}
        exec("from torch_rs.jit import *", actual_namespace)
        exec("from torch.jit import *", expected_namespace)
        self.assertEqual(
            {name for name in actual_namespace if not name.startswith("__")},
            wildcard_supported,
        )
        for name in wildcard_supported:
            self.assertIs(actual_namespace[name], getattr(actual_jit, name))
            self.assertIs(expected_namespace[name], getattr(expected_jit, name))

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("jit", namespace)
            for name in ("enable_onednn_fusion", "onednn_fusion_enabled"):
                self.assertNotIn(name, namespace)
                self.assertFalse(hasattr(module, name))

        for name in ("enable_onednn_fusion", "onednn_fusion_enabled"):
            actual = getattr(actual_jit, name)
            expected = getattr(expected_jit, name)
            self.assertIs(copy.copy(actual), actual)
            self.assertIs(copy.copy(expected), expected)
            self.assertIs(copy.deepcopy(actual), actual)
            self.assertIs(copy.deepcopy(expected), expected)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(name=name, protocol=protocol):
                    self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                    self.assertIs(
                        pickle.loads(pickle.dumps(expected, protocol)), expected
                    )
                    self.assertEqual(
                        self.pickle_shape(actual, protocol),
                        self.pickle_shape(expected, protocol),
                    )

    def test_call_shape_errors_are_non_mutating_and_match(self):
        actual_setter = torch.jit.enable_onednn_fusion
        expected_setter = reference_torch.jit.enable_onednn_fusion
        actual_getter = torch.jit.onednn_fusion_enabled
        expected_getter = reference_torch.jit.onednn_fusion_enabled
        actual_setter(True)
        expected_setter(True)
        cases = (
            (lambda: actual_setter(), lambda: expected_setter()),
            (
                lambda: actual_setter(True, False),
                lambda: expected_setter(True, False),
            ),
            (
                lambda: actual_setter(value=True),
                lambda: expected_setter(value=True),
            ),
            (
                lambda: actual_setter(True, enabled=False),
                lambda: expected_setter(True, enabled=False),
            ),
            (lambda: actual_getter(None), lambda: expected_getter(None)),
            (
                lambda: actual_getter(enabled=True),
                lambda: expected_getter(enabled=True),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)
                self.assertIs(actual_getter(), True)
                self.assertIs(expected_getter(), True)

    def test_torchscript_compilation_and_fusion_execution_remain_unsupported(self):
        self.assertTrue(callable(torch.jit.enable_onednn_fusion))
        self.assertTrue(callable(reference_torch.jit.enable_onednn_fusion))
        expected_public = {
            name for name in vars(reference_torch.jit) if not name.startswith("_")
        }
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
                self.assertIn(name, expected_public)
                self.assertFalse(hasattr(torch.jit, name))
        self.assertFalse(hasattr(torch, "compile"))


if __name__ == "__main__":
    unittest.main()
