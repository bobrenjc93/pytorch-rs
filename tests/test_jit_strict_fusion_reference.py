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
import warnings
from unittest import mock

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class JitStrictFusionReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "jit.strict_fusion differentials require pinned PyTorch 2.13.0"
            )

    def make_context(self, module):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            context = module.jit.strict_fusion()
        return context, [
            (warning.category, str(warning.message), warning.message.args)
            for warning in caught
        ]

    def context_outcome(self, module):
        context, constructor_warnings = self.make_context(module)
        states = []

        def run(expected_grad_state):
            before = module.is_grad_enabled()
            with context as entered:
                during = module.is_grad_enabled()
                with context as nested_entered:
                    nested = module.is_grad_enabled()
                after_nested = module.is_grad_enabled()
            after = module.is_grad_enabled()
            states.append(
                (
                    before,
                    entered,
                    during,
                    nested_entered,
                    nested,
                    after_nested,
                    after,
                    expected_grad_state,
                )
            )

        run(True)
        with module.no_grad():
            run(False)
        run(True)

        marker = RuntimeError("strict fusion body failed")
        try:
            with context:
                raise marker
        except RuntimeError as error:
            propagated_identity = error is marker
        else:
            propagated_identity = False

        return constructor_warnings, states, propagated_identity

    def threaded_outcome(self, module):
        context, _ = self.make_context(module)
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                mode = module.no_grad() if index % 2 else contextlib.nullcontext()
                initial = module.is_grad_enabled()
                with mode:
                    before = module.is_grad_enabled()
                    barrier.wait(timeout=10)
                    with context as entered:
                        during = module.is_grad_enabled()
                    after = module.is_grad_enabled()
                final = module.is_grad_enabled()
                results[index] = (initial, before, entered, during, after, final)
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
        return results, errors

    def pickle_shape(self, value, protocol):
        shape = []
        for opcode, argument, _ in pickletools.genops(
            pickle.dumps(value, protocol=protocol)
        ):
            if opcode.name == "FRAME":
                argument = "<frame length>"
            elif isinstance(argument, str):
                argument = argument.replace("torch_rs", "torch")
            shape.append((opcode.name, argument))
        return shape

    def assert_error_matches(self, actual_call, expected_call):
        with warnings.catch_warnings(record=True) as actual_warnings:
            warnings.simplefilter("always")
            with self.assertRaises(Exception) as actual_raised:
                actual_call()
        with warnings.catch_warnings(record=True) as expected_warnings:
            warnings.simplefilter("always")
            with self.assertRaises(Exception) as expected_raised:
                expected_call()
        self.assertEqual(actual_warnings, [])
        self.assertEqual(expected_warnings, [])
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def test_eager_warning_context_nesting_and_exception_behavior_match(self):
        self.assertEqual(
            self.context_outcome(torch),
            self.context_outcome(reference_torch),
        )
        self.assertEqual(
            self.threaded_outcome(torch),
            self.threaded_outcome(reference_torch),
        )

    def test_signature_documentation_and_ownership_match(self):
        actual_jit = importlib.import_module("torch_rs.jit")
        expected_jit = importlib.import_module("torch.jit")
        actual = actual_jit.strict_fusion
        expected = expected_jit.strict_fusion

        self.assertIs(torch.jit, actual_jit)
        self.assertIs(reference_torch.jit, expected_jit)
        self.assertIs(type(actual), type(expected))
        self.assertEqual(actual.__bases__, expected.__bases__)
        self.assertEqual(str(inspect.signature(actual)), str(inspect.signature(expected)))
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(actual.__module__.replace("torch_rs", "torch"), expected.__module__)
        self.assertIs(inspect.getmodule(actual), actual_jit)
        self.assertIs(inspect.getmodule(expected), expected_jit)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(set(actual.__dict__), set(expected.__dict__))

        for name in ("__init__", "__enter__", "__exit__"):
            with self.subTest(name=name):
                actual_method = getattr(actual, name)
                expected_method = getattr(expected, name)
                self.assertIs(type(actual_method), types.FunctionType)
                self.assertIs(type(expected_method), types.FunctionType)
                self.assertEqual(
                    str(inspect.signature(actual_method)),
                    str(inspect.signature(expected_method)),
                )
                self.assertEqual(
                    actual_method.__annotations__, expected_method.__annotations__
                )
                self.assertEqual(
                    typing.get_type_hints(actual_method),
                    typing.get_type_hints(expected_method),
                )
                self.assertEqual(actual_method.__name__, expected_method.__name__)
                self.assertEqual(actual_method.__qualname__, expected_method.__qualname__)
                self.assertEqual(
                    actual_method.__module__.replace("torch_rs", "torch"),
                    expected_method.__module__,
                )
                self.assertIs(inspect.getmodule(actual_method), actual_jit)
                self.assertIs(inspect.getmodule(expected_method), expected_jit)
                self.assertEqual(actual_method.__doc__, expected_method.__doc__)
                self.assertEqual(actual_method.__defaults__, expected_method.__defaults__)
                self.assertEqual(actual_method.__kwdefaults__, expected_method.__kwdefaults__)
                self.assertEqual(actual_method.__dict__, expected_method.__dict__)

    def test_exports_match_the_newly_supported_scope(self):
        actual_jit = torch.jit
        expected_jit = reference_torch.jit
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
            actual_jit.__all__,
            [
                name for name in expected_jit.__all__ if name in wildcard_supported
            ],
        )
        self.assertEqual(
            {name for name in vars(actual_jit) if not name.startswith("_")},
            {*wildcard_supported, "is_scripting", "is_tracing"},
        )

        actual_namespace = {}
        expected_namespace = {}
        exec("from torch_rs.jit import *", actual_namespace)
        exec("from torch.jit import *", expected_namespace)
        self.assertEqual(
            {name for name in actual_namespace if not name.startswith("__")},
            wildcard_supported,
        )
        self.assertIs(actual_namespace["strict_fusion"], actual_jit.strict_fusion)
        self.assertIs(expected_namespace["strict_fusion"], expected_jit.strict_fusion)

        for module in (torch, reference_torch):
            self.assertFalse(hasattr(module, "strict_fusion"))
            self.assertNotIn("strict_fusion", module.__all__)
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("strict_fusion", namespace)

    def test_copying_and_pickling_match(self):
        actual_type = torch.jit.strict_fusion
        expected_type = reference_torch.jit.strict_fusion
        actual, _ = self.make_context(torch)
        expected, _ = self.make_context(reference_torch)
        actual.payload = {"items": [1, 2]}
        expected.payload = {"items": [1, 2]}

        for context, context_type in (
            (actual, actual_type),
            (expected, expected_type),
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                shallow = copy.copy(context)
                deep = copy.deepcopy(context)
            self.assertEqual(caught, [])
            self.assertIsNot(shallow, context)
            self.assertIs(type(shallow), context_type)
            self.assertIs(shallow.payload, context.payload)
            self.assertIsNot(deep, context)
            self.assertIs(type(deep), context_type)
            self.assertEqual(deep.__dict__, context.__dict__)
            self.assertIsNot(deep.payload, context.payload)
            self.assertIs(copy.copy(context_type), context_type)
            self.assertIs(copy.deepcopy(context_type), context_type)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertEqual(
                    self.pickle_shape(actual_type, protocol),
                    self.pickle_shape(expected_type, protocol),
                )
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )
                self.assertIs(
                    pickle.loads(pickle.dumps(actual_type, protocol)), actual_type
                )
                self.assertIs(
                    pickle.loads(pickle.dumps(expected_type, protocol)), expected_type
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    actual_restored = pickle.loads(pickle.dumps(actual, protocol))
                    expected_restored = pickle.loads(pickle.dumps(expected, protocol))
                self.assertEqual(caught, [])
                self.assertIs(type(actual_restored), actual_type)
                self.assertIs(type(expected_restored), expected_type)
                self.assertEqual(actual_restored.__dict__, expected_restored.__dict__)

    def test_argument_errors_and_scripting_warning_gate_match(self):
        actual = torch.jit.strict_fusion
        expected = reference_torch.jit.strict_fusion
        cases = (
            (lambda: actual(None), lambda: expected(None)),
            (lambda: actual(None, None), lambda: expected(None, None)),
            (lambda: actual(enabled=True), lambda: expected(enabled=True)),
            (
                lambda: actual(None, enabled=True),
                lambda: expected(None, enabled=True),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

        for module in (torch, reference_torch):
            with mock.patch.object(
                module._jit_internal, "is_scripting", return_value=True
            ) as is_scripting:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    context = module.jit.strict_fusion()
            is_scripting.assert_called_once_with()
            self.assertEqual(caught, [])
            with context as entered:
                self.assertIsNone(entered)

    def test_scripting_tracing_and_fusion_execution_remain_unsupported(self):
        self.assertTrue(callable(reference_torch.jit.script))
        self.assertTrue(callable(reference_torch.jit.trace))
        self.assertFalse(hasattr(torch.jit, "script"))
        self.assertFalse(hasattr(torch.jit, "trace"))
        self.assertFalse(hasattr(torch, "compile"))
        self.assertIs(torch.jit.is_scripting(), False)
        self.assertIs(torch.jit.is_tracing(), False)


if __name__ == "__main__":
    unittest.main()
