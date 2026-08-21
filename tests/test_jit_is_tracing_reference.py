import contextlib
import copy
import importlib
import inspect
import pickle
import pickletools
import sys
import threading
import types
import typing
import unittest
import warnings

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class JitIsTracingReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "jit.is_tracing differentials require pinned PyTorch 2.13.0"
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
        function = module.jit.is_tracing

        def query_outcome():
            before = module.is_grad_enabled()
            result = function()
            after = module.is_grad_enabled()
            return before, result is False, after

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

    def test_supported_eager_threaded_and_grad_states_match_pytorch_2_13(self):
        self.assertEqual(
            self.supported_state_outcome(torch),
            self.supported_state_outcome(reference_torch),
        )

    def test_signature_documentation_and_trace_module_identity_match(self):
        actual_jit = importlib.import_module("torch_rs.jit")
        expected_jit = importlib.import_module("torch.jit")
        actual_trace = importlib.import_module("torch_rs.jit._trace")
        expected_trace = importlib.import_module("torch.jit._trace")
        actual = actual_jit.is_tracing
        expected = expected_jit.is_tracing

        self.assertIs(torch.jit, actual_jit)
        self.assertIs(reference_torch.jit, expected_jit)
        self.assertIs(actual_jit._trace, actual_trace)
        self.assertIs(expected_jit._trace, expected_trace)
        self.assertIs(sys.modules["torch_rs.jit._trace"], actual_trace)
        self.assertIs(sys.modules["torch.jit._trace"], expected_trace)
        self.assertIs(actual, actual_trace.is_tracing)
        self.assertIs(expected, expected_trace.is_tracing)
        self.assertEqual(actual_trace.__doc__, expected_trace.__doc__)
        self.assertEqual(
            actual_trace.__name__.replace("torch_rs", "torch"),
            expected_trace.__name__,
        )
        self.assertEqual(
            actual_trace.__package__.replace("torch_rs", "torch"),
            expected_trace.__package__,
        )
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual)), str(inspect.signature(expected))
        )
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(typing.get_type_hints(actual), typing.get_type_hints(expected))
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertIs(inspect.getmodule(actual), actual_trace)
        self.assertIs(inspect.getmodule(expected), expected_trace)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )

    def test_imports_exports_copy_and_pickle_match_the_supported_scope(self):
        actual_jit = torch.jit
        expected_jit = reference_torch.jit
        actual_trace = actual_jit._trace
        expected_trace = expected_jit._trace
        actual = actual_jit.is_tracing
        expected = expected_jit.is_tracing
        wildcard_supported = {
            "annotate",
            "export",
            "ignore",
            "isinstance",
            "script_if_tracing",
            "unused",
        }
        public_supported = {
            *wildcard_supported,
            "is_scripting",
            "is_tracing",
        }

        self.assertEqual(
            actual_jit.__all__,
            [
                name
                for name in expected_jit.__all__
                if name in wildcard_supported
            ],
        )
        self.assertNotIn("is_tracing", actual_jit.__all__)
        self.assertNotIn("is_tracing", expected_jit.__all__)
        self.assertEqual(
            {name for name in vars(actual_jit) if not name.startswith("_")},
            public_supported,
        )
        self.assertFalse(hasattr(actual_trace, "__all__"))
        self.assertFalse(hasattr(expected_trace, "__all__"))
        self.assertEqual(
            {name for name in vars(actual_trace) if not name.startswith("_")},
            {"is_tracing"},
        )
        self.assertIn("is_tracing", vars(expected_trace))
        self.assertEqual(
            torch.__all__.count("jit"), reference_torch.__all__.count("jit")
        )
        self.assertEqual(
            torch.__all__.count("is_tracing"),
            reference_torch.__all__.count("is_tracing"),
        )

        actual_package_import = {}
        expected_package_import = {}
        exec("from torch_rs.jit import is_tracing", actual_package_import)
        exec("from torch.jit import is_tracing", expected_package_import)
        self.assertIs(actual_package_import["is_tracing"], actual)
        self.assertIs(expected_package_import["is_tracing"], expected)

        actual_module_import = {}
        expected_module_import = {}
        exec("from torch_rs.jit._trace import is_tracing", actual_module_import)
        exec("from torch.jit._trace import is_tracing", expected_module_import)
        self.assertIs(actual_module_import["is_tracing"], actual)
        self.assertIs(expected_module_import["is_tracing"], expected)

        actual_namespace = {}
        expected_namespace = {}
        exec("from torch_rs.jit import *", actual_namespace)
        exec("from torch.jit import *", expected_namespace)
        self.assertEqual(
            {name for name in actual_namespace if not name.startswith("__")},
            wildcard_supported,
        )
        self.assertNotIn("is_tracing", actual_namespace)
        self.assertNotIn("is_tracing", expected_namespace)

        actual_trace_namespace = {}
        expected_trace_namespace = {}
        exec("from torch_rs.jit._trace import *", actual_trace_namespace)
        exec("from torch.jit._trace import *", expected_trace_namespace)
        self.assertEqual(
            {
                name
                for name in actual_trace_namespace
                if not name.startswith("__")
            },
            {"is_tracing"},
        )
        self.assertIs(actual_trace_namespace["is_tracing"], actual)
        self.assertIs(expected_trace_namespace["is_tracing"], expected)

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("jit", namespace)
            self.assertNotIn("is_tracing", namespace)
            self.assertFalse(hasattr(module, "is_tracing"))

        self.assertIs(copy.copy(actual), actual)
        self.assertIs(copy.copy(expected), expected)
        self.assertIs(copy.deepcopy(actual), actual)
        self.assertIs(copy.deepcopy(expected), expected)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(pickle.loads(pickle.dumps(expected, protocol)), expected)
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.jit.is_tracing
        expected = reference_torch.jit.is_tracing
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

    def test_reference_trace_bounds_the_unsupported_true_state(self):
        actual_is_tracing = torch.jit.is_tracing
        expected_is_tracing = reference_torch.jit.is_tracing
        actual_states = [actual_is_tracing()]
        expected_states = [expected_is_tracing()]

        def probe(value):
            actual_states.append(actual_is_tracing())
            expected_states.append(expected_is_tracing())
            return value + 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            traced = reference_torch.jit.trace(
                probe,
                reference_torch.tensor(1.0),
                check_trace=False,
            )
        result = traced(reference_torch.tensor(2.0))
        actual_states.append(actual_is_tracing())
        expected_states.append(expected_is_tracing())

        self.assertEqual(result.item(), 3.0)
        self.assertEqual(actual_states, [False, False, False])
        self.assertEqual(expected_states, [False, True, False])
        for state in (*actual_states, *expected_states):
            self.assertIs(type(state), bool)

    def test_tracing_scripting_and_compilation_remain_unsupported(self):
        self.assertTrue(callable(reference_torch.jit.trace))
        self.assertTrue(callable(reference_torch.jit.script))
        self.assertFalse(hasattr(torch, "compile"))
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
        self.assertFalse(hasattr(torch.jit._trace, "trace"))
        self.assertFalse(hasattr(torch.jit._trace, "trace_module"))


if __name__ == "__main__":
    unittest.main()
