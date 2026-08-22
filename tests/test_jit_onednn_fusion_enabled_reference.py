import contextlib
import copy
import importlib
import inspect
import pickle
import pickletools
import threading
import types
import typing
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class JitOnednnFusionEnabledReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "jit.onednn_fusion_enabled differentials require pinned "
                "PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def supported_state_outcome(self, module):
        function = module.jit.onednn_fusion_enabled

        def query_outcome():
            before = module.is_grad_enabled()
            first = function()
            middle = module.is_grad_enabled()
            second = function()
            after = module.is_grad_enabled()
            return (
                before,
                type(first) is bool,
                first,
                middle,
                type(second) is bool,
                second,
                after,
            )

        states = [query_outcome()]
        with module.no_grad():
            states.append(query_outcome())
            with module.no_grad():
                states.append(query_outcome())
            states.append(query_outcome())
        states.append(query_outcome())

        worker_count = 8
        barrier = threading.Barrier(worker_count)
        worker_states = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = module.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    worker_states[index] = query_outcome()
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))

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
        return states, worker_states

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

    def test_supported_false_threaded_and_grad_states_match_pytorch_2_13(self):
        getter = reference_torch.jit.onednn_fusion_enabled
        setter = reference_torch.jit.enable_onednn_fusion
        original_enabled = getter()
        try:
            self.assertIs(setter(False), None)
            self.assertEqual(
                self.supported_state_outcome(torch),
                self.supported_state_outcome(reference_torch),
            )
        finally:
            setter(original_enabled)

        self.assertIs(torch.jit.onednn_fusion_enabled(), False)
        self.assertIs(getter(), original_enabled)

    def test_reference_only_setter_bounds_the_unsupported_true_state(self):
        actual = torch.jit.onednn_fusion_enabled
        expected = reference_torch.jit.onednn_fusion_enabled
        setter = reference_torch.jit.enable_onednn_fusion
        original_enabled = expected()
        try:
            self.assertIs(setter(False), None)
            self.assertIs(actual(), False)
            self.assertIs(expected(), False)

            with torch.no_grad(), reference_torch.no_grad():
                self.assertIs(setter(True), None)
                self.assertIs(torch.is_grad_enabled(), False)
                self.assertIs(reference_torch.is_grad_enabled(), False)
                self.assertIs(actual(), False)
                reference_enabled = expected()
                if not reference_enabled:
                    self.skipTest(
                        "reference PyTorch build cannot enable oneDNN Graph fusion"
                    )
                self.assertIs(reference_enabled, True)

            self.assertIs(setter(False), None)
            self.assertIs(actual(), False)
            self.assertIs(expected(), False)
        finally:
            setter(original_enabled)

        self.assertIs(actual(), False)
        self.assertIs(expected(), original_enabled)

    def test_signature_documentation_and_identity_match(self):
        actual_jit = importlib.import_module("torch_rs.jit")
        expected_jit = importlib.import_module("torch.jit")
        actual = actual_jit.onednn_fusion_enabled
        expected = expected_jit.onednn_fusion_enabled

        self.assertIs(torch.jit, actual_jit)
        self.assertIs(reference_torch.jit, expected_jit)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual)),
            str(inspect.signature(expected)),
        )
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(typing.get_type_hints(actual), typing.get_type_hints(expected))
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
        self.assertEqual(actual_jit.__doc__, expected_jit.__doc__)

    def test_exports_imports_copying_and_pickling_match_supported_scope(self):
        actual_jit = torch.jit
        expected_jit = reference_torch.jit
        actual = actual_jit.onednn_fusion_enabled
        expected = expected_jit.onednn_fusion_enabled
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
            actual_jit.__all__,
            [
                name
                for name in expected_jit.__all__
                if name in wildcard_supported
            ],
        )
        self.assertEqual(
            {name for name in vars(actual_jit) if not name.startswith("_")},
            {*wildcard_supported, "is_scripting", "is_tracing"},
        )

        actual_explicit = {}
        expected_explicit = {}
        exec(
            "from torch_rs.jit import onednn_fusion_enabled",
            actual_explicit,
        )
        exec(
            "from torch.jit import onednn_fusion_enabled",
            expected_explicit,
        )
        self.assertIs(actual_explicit["onednn_fusion_enabled"], actual)
        self.assertIs(expected_explicit["onednn_fusion_enabled"], expected)

        actual_wildcard = {}
        expected_wildcard = {}
        exec("from torch_rs.jit import *", actual_wildcard)
        exec("from torch.jit import *", expected_wildcard)
        self.assertEqual(
            {
                name
                for name in actual_wildcard
                if not name.startswith("__")
            },
            wildcard_supported,
        )
        self.assertIs(actual_wildcard["onednn_fusion_enabled"], actual)
        self.assertIs(expected_wildcard["onednn_fusion_enabled"], expected)

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("onednn_fusion_enabled", namespace)
            self.assertFalse(hasattr(module, "onednn_fusion_enabled"))
        self.assertEqual(
            torch.__all__.count("onednn_fusion_enabled"),
            reference_torch.__all__.count("onednn_fusion_enabled"),
        )

        for function in (actual, expected):
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(pickle.loads(pickle.dumps(expected, protocol)), expected)
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.jit.onednn_fusion_enabled
        expected = reference_torch.jit.onednn_fusion_enabled
        cases = (
            (lambda: actual(None), lambda: expected(None)),
            (lambda: actual(None, None), lambda: expected(None, None)),
            (
                lambda: actual(enabled=True),
                lambda: expected(enabled=True),
            ),
            (
                lambda: actual(None, enabled=True),
                lambda: expected(None, enabled=True),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)
        self.assertIs(actual(**{}), False)
        self.assertIs(expected(**{}), False)

    def test_setter_torchscript_and_fusion_execution_are_outside_scope(self):
        self.assertTrue(callable(reference_torch.jit.enable_onednn_fusion))
        self.assertIn("enable_onednn_fusion", reference_torch.jit.__all__)
        self.assertFalse(hasattr(torch.jit, "enable_onednn_fusion"))
        self.assertNotIn("enable_onednn_fusion", torch.jit.__all__)
        self.assertTrue(hasattr(reference_torch._C, "_jit_llga_enabled"))
        self.assertTrue(hasattr(reference_torch._C, "_jit_set_llga_enabled"))
        self.assertFalse(hasattr(torch._C, "_jit_llga_enabled"))
        self.assertFalse(hasattr(torch._C, "_jit_set_llga_enabled"))

        expected_public = {
            name for name in vars(reference_torch.jit) if not name.startswith("_")
        }
        for name in (
            "CompilationUnit",
            "ScriptFunction",
            "ScriptModule",
            "script",
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
