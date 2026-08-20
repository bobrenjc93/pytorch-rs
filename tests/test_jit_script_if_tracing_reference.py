import copy
import importlib
import inspect
import pickle
import pickletools
import re
import types
import typing
import unittest
import warnings

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@torch.jit.script_if_tracing
def _actual_picklable_script_if_tracing_function(value):
    return value


if reference_torch is not None:

    @reference_torch.jit.script_if_tracing
    def _expected_picklable_script_if_tracing_function(value):
        return value


def _reference_control_flow(value):
    if value.sum() > 0:
        return value * 2
    return value - 2


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class JitScriptIfTracingReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "jit.script_if_tracing differentials require pinned PyTorch 2.13.0"
            )

    def normalize_error(self, error):
        return (
            type(error),
            re.sub(r"0x[0-9a-fA-F]+", "0x<address>", str(error)),
            tuple(
                re.sub(r"0x[0-9a-fA-F]+", "0x<address>", value)
                if isinstance(value, str)
                else value
                for value in error.args
            ),
        )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(
            self.normalize_error(actual_raised.exception),
            self.normalize_error(expected_raised.exception),
        )

    def pickle_shape(self, value, protocol):
        shape = []
        for opcode, argument, _ in pickletools.genops(
            pickle.dumps(value, protocol=protocol)
        ):
            if opcode.name == "FRAME":
                argument = "<frame length>"
            elif isinstance(argument, str):
                argument = argument.replace("torch_rs", "torch")
                argument = argument.replace("_actual_", "_expected_")
            shape.append((opcode.name, argument))
        return shape

    def eager_outcome(self, module):
        default = object()
        first = object()
        second = object()
        required = object()
        named = object()
        result = object()
        calls = []

        def function(
            positional,
            /,
            optional=default,
            *args,
            keyword,
            optional_keyword=default,
            **kwargs,
        ):
            calls.append(
                (positional, optional, args, keyword, optional_keyword, kwargs)
            )
            return result

        function.custom_attribute = named
        wrapped = module.jit.script_if_tracing(function)
        returned = wrapped(first, second, named, keyword=required, named=named)
        call = calls[0]

        expected_error = RuntimeError("same exception")

        def raises():
            raise expected_error

        wrapped_raises = module.jit.script_if_tracing(raises)
        with self.assertRaises(RuntimeError) as raised:
            wrapped_raises()

        return (
            wrapped is not function,
            returned is result,
            len(calls),
            call[0] is first,
            call[1] is second,
            call[2] == (named,),
            call[2][0] is named,
            call[3] is required,
            call[4] is default,
            set(call[5]) == {"named"},
            call[5]["named"] is named,
            wrapped.custom_attribute is named,
            wrapped.__wrapped__ is function,
            getattr(wrapped, "__original_fn") is function,
            getattr(wrapped, "__script_if_tracing_wrapper") is True,
            inspect.signature(wrapped) == inspect.signature(function),
            raised.exception is expected_error,
            raised.exception.args,
        )

    def binding_outcome(self, module):
        class Example:
            @module.jit.script_if_tracing
            def method(self, value, *, offset=1):
                return self, value + offset

            @staticmethod
            @module.jit.script_if_tracing
            def static_method(value):
                return value + 2

            @classmethod
            @module.jit.script_if_tracing
            def class_method(cls, value):
                return cls, value + 3

        instance = Example()
        raw_method = Example.__dict__["method"]
        bound_method = instance.method
        method_owner, method_value = bound_method(4, offset=5)
        class_owner, class_value = Example.class_method(4)
        return (
            bound_method.__self__ is instance,
            bound_method.__func__ is raw_method,
            getattr(raw_method, "__original_fn") is raw_method.__wrapped__,
            getattr(raw_method, "__script_if_tracing_wrapper") is True,
            method_owner is instance,
            method_value,
            Example.static_method(4),
            class_owner is Example,
            class_value,
        )

    def test_eager_forwarding_metadata_and_binding_match_pytorch_2_13(self):
        self.assertEqual(
            self.eager_outcome(torch), self.eager_outcome(reference_torch)
        )
        self.assertEqual(
            self.binding_outcome(torch), self.binding_outcome(reference_torch)
        )

    def test_signature_documentation_and_ownership_match_pytorch_2_13(self):
        actual_jit = importlib.import_module("torch_rs.jit")
        expected_jit = importlib.import_module("torch.jit")
        actual_trace = importlib.import_module("torch_rs.jit._trace")
        expected_trace = importlib.import_module("torch.jit._trace")
        actual = actual_jit.script_if_tracing
        expected = expected_jit.script_if_tracing
        actual_helper = actual_jit._script_if_tracing
        expected_helper = expected_jit._script_if_tracing

        self.assertIs(actual_jit._trace, actual_trace)
        self.assertIs(expected_jit._trace, expected_trace)
        self.assertIs(actual_helper, actual_trace._script_if_tracing)
        self.assertIs(expected_helper, expected_trace._script_if_tracing)
        self.assertIsNot(actual, actual_helper)
        self.assertIsNot(expected, expected_helper)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(str(inspect.signature(actual)), str(inspect.signature(expected)))
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(typing.get_type_hints(actual), typing.get_type_hints(expected))
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
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

        self.assertEqual(
            str(inspect.signature(actual_helper)),
            str(inspect.signature(expected_helper)),
        )
        self.assertEqual(set(actual_helper.__annotations__), {"fn", "return"})
        self.assertEqual(
            repr(actual_helper.__annotations__).replace("torch_rs", "torch"),
            repr(expected_helper.__annotations__),
        )
        self.assertEqual(actual_helper.__name__, expected_helper.__name__)
        self.assertEqual(actual_helper.__qualname__, expected_helper.__qualname__)
        self.assertEqual(
            actual_helper.__module__.replace("torch_rs", "torch"),
            expected_helper.__module__,
        )
        self.assertIs(inspect.getmodule(actual_helper), actual_trace)
        self.assertIs(inspect.getmodule(expected_helper), expected_trace)
        self.assertEqual(actual_helper.__doc__, expected_helper.__doc__)
        self.assertEqual(actual_helper.__defaults__, expected_helper.__defaults__)
        self.assertEqual(actual_helper.__kwdefaults__, expected_helper.__kwdefaults__)
        self.assertEqual(actual_helper.__dict__, expected_helper.__dict__)

    def test_exports_copy_and_pickle_match_the_supported_scope(self):
        actual_jit = torch.jit
        expected_jit = reference_torch.jit
        actual = actual_jit.script_if_tracing
        expected = expected_jit.script_if_tracing
        actual_helper = actual_jit._script_if_tracing
        expected_helper = expected_jit._script_if_tracing
        wildcard_supported = {
            "annotate",
            "export",
            "ignore",
            "script_if_tracing",
            "unused",
        }
        public_supported = {*wildcard_supported, "is_scripting", "is_tracing"}

        self.assertEqual(
            actual_jit.__all__,
            [name for name in expected_jit.__all__ if name in wildcard_supported],
        )
        self.assertIn("script_if_tracing", actual_jit.__all__)
        self.assertIn("script_if_tracing", expected_jit.__all__)
        self.assertEqual(
            {name for name in vars(actual_jit) if not name.startswith("_")},
            public_supported,
        )
        self.assertEqual(
            {name for name in vars(actual_jit._trace) if not name.startswith("_")},
            {"is_tracing"},
        )

        actual_namespace = {}
        expected_namespace = {}
        exec("from torch_rs.jit import *", actual_namespace)
        exec("from torch.jit import *", expected_namespace)
        self.assertEqual(
            {name for name in actual_namespace if not name.startswith("__")},
            wildcard_supported,
        )
        self.assertIs(actual_namespace["script_if_tracing"], actual)
        self.assertIs(expected_namespace["script_if_tracing"], expected)

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("jit", namespace)
            self.assertNotIn("script_if_tracing", namespace)
            self.assertFalse(hasattr(module, "script_if_tracing"))

        pairs = (
            (actual, expected),
            (actual_helper, expected_helper),
            (
                _actual_picklable_script_if_tracing_function,
                _expected_picklable_script_if_tracing_function,
            ),
        )
        for actual_value, expected_value in pairs:
            self.assertIs(copy.copy(actual_value), actual_value)
            self.assertIs(copy.copy(expected_value), expected_value)
            self.assertIs(copy.deepcopy(actual_value), actual_value)
            self.assertIs(copy.deepcopy(expected_value), expected_value)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(value=actual_value, protocol=protocol):
                    self.assertIs(
                        pickle.loads(pickle.dumps(actual_value, protocol)),
                        actual_value,
                    )
                    self.assertIs(
                        pickle.loads(pickle.dumps(expected_value, protocol)),
                        expected_value,
                    )
                    self.assertEqual(
                        self.pickle_shape(actual_value, protocol),
                        self.pickle_shape(expected_value, protocol),
                    )

    def test_invalid_calls_and_targets_match_pytorch_2_13(self):
        actual = torch.jit.script_if_tracing
        expected = reference_torch.jit.script_if_tracing
        calls = (
            lambda function: function(),
            lambda function: function(lambda: None, lambda: None),
            lambda function: function(function=lambda: None),
            lambda function: function(lambda: None, fn=lambda: None),
        )
        for case, call in enumerate(calls):
            with self.subTest(call_case=case):
                self.assert_error_matches(
                    lambda: call(actual),
                    lambda: call(expected),
                )

        targets = (
            (lambda: None),
            (lambda: 1),
            (lambda: property()),
        )
        for case, make_target in enumerate(targets):
            actual_target = make_target()
            expected_target = make_target()
            actual_wrapped = actual(actual_target)
            expected_wrapped = expected(expected_target)
            with self.subTest(target_case=case):
                self.assertIs(
                    getattr(actual_wrapped, "__original_fn"), actual_target
                )
                self.assertIs(
                    getattr(expected_wrapped, "__original_fn"), expected_target
                )
                self.assertEqual(
                    getattr(actual_wrapped, "__script_if_tracing_wrapper"),
                    getattr(expected_wrapped, "__script_if_tracing_wrapper"),
                )
                self.assert_error_matches(actual_wrapped, expected_wrapped)

        actual_len = actual(len)
        expected_len = expected(len)
        self.assertEqual(actual_len([1, 2, 3]), expected_len([1, 2, 3]))
        self.assert_error_matches(actual_len, expected_len)

    def test_reference_only_tracing_bounds_the_unsupported_compiled_path(self):
        actual_is_tracing = torch.jit.is_tracing
        expected_is_tracing = reference_torch.jit.is_tracing
        actual_states = [actual_is_tracing()]
        expected_states = [expected_is_tracing()]

        def actual_function(value):
            actual_states.append(actual_is_tracing())
            return value

        actual_wrapped = torch.jit.script_if_tracing(actual_function)
        sentinel = object()
        self.assertIs(actual_wrapped(sentinel), sentinel)

        expected_wrapped = reference_torch.jit.script_if_tracing(
            _reference_control_flow
        )

        def reference_probe(value):
            expected_states.append(expected_is_tracing())
            return expected_wrapped(value)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            traced = reference_torch.jit.trace(
                reference_probe,
                reference_torch.tensor([1.0]),
                check_trace=False,
            )
        result = traced(reference_torch.tensor([-1.0]))
        actual_states.append(actual_is_tracing())
        expected_states.append(expected_is_tracing())

        self.assertEqual(result.tolist(), [-3.0])
        self.assertEqual(actual_states, [False, False, False])
        self.assertEqual(expected_states, [False, True, False])
        for state in (*actual_states, *expected_states):
            self.assertIs(type(state), bool)

        self.assertTrue(callable(reference_torch.jit.script))
        self.assertTrue(callable(reference_torch.jit.trace))
        self.assertFalse(hasattr(torch.jit, "script"))
        self.assertFalse(hasattr(torch.jit, "trace"))
        self.assertIs(torch.jit.is_tracing, actual_is_tracing)


if __name__ == "__main__":
    unittest.main()
