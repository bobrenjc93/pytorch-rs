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


def _actual_picklable_script_if_tracing_function(value):
    return value + 1


_actual_picklable_script_if_tracing_function = torch.jit.script_if_tracing(
    _actual_picklable_script_if_tracing_function
)


def _expected_picklable_script_if_tracing_function(value):
    return value + 1


def _reference_control_flow(value):
    if value.sum() > 0:
        return value * 2
    return value - 2


if reference_torch is not None:
    _expected_picklable_script_if_tracing_function = (
        reference_torch.jit.script_if_tracing(
            _expected_picklable_script_if_tracing_function
        )
    )


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
            shape.append((opcode.name, argument))
        return shape

    def annotation_shape(self, function):
        return {
            name: repr(annotation).replace("torch_rs", "torch")
            for name, annotation in function.__annotations__.items()
        }

    def eager_outcome(self, module):
        sentinel = object()
        result = object()
        calls = []

        def function(
            value: int, offset: int = 2, *, option: object = None
        ) -> object:
            """Original function documentation."""
            calls.append((value, offset, option, module.jit.is_tracing()))
            return result

        function.custom_attribute = sentinel
        wrapper = module.jit.script_if_tracing(function)
        returned = wrapper(3, option="set")
        return (
            wrapper is not function,
            returned is result,
            calls == [(3, 2, "set", False)],
            str(inspect.signature(wrapper)),
            wrapper.__name__,
            wrapper.__qualname__,
            wrapper.__module__,
            wrapper.__doc__,
            wrapper.__annotations__,
            wrapper.__defaults__,
            wrapper.__kwdefaults__,
            wrapper.custom_attribute is sentinel,
            wrapper.__wrapped__ is function,
            getattr(wrapper, "__original_fn") is function,
            getattr(wrapper, "__script_if_tracing_wrapper") is True,
            inspect.unwrap(wrapper) is function,
            copy.copy(wrapper) is wrapper,
            copy.deepcopy(wrapper) is wrapper,
        )

    def binding_outcome(self, module):
        class Example:
            @module.jit.script_if_tracing
            def method(self, value=1):
                return self, value

            @staticmethod
            @module.jit.script_if_tracing
            def static_method(value=2):
                return value

            @classmethod
            @module.jit.script_if_tracing
            def class_method(cls, value=3):
                return cls, value

        instance = Example()
        raw_method = Example.__dict__["method"]
        raw_static_method = Example.__dict__["static_method"].__func__
        raw_class_method = Example.__dict__["class_method"].__func__
        bound_result = instance.method(4)
        class_result = Example.class_method(6)
        direct_bound_wrapper = module.jit.script_if_tracing(instance.method)
        direct_bound_result = direct_bound_wrapper(7)
        return (
            instance.method.__self__ is instance,
            instance.method.__func__ is raw_method,
            bound_result[0] is instance,
            bound_result[1],
            Example.static_method(5),
            instance.static_method(),
            class_result[0] is Example,
            class_result[1],
            tuple(
                (
                    wrapper.__wrapped__ is getattr(wrapper, "__original_fn"),
                    getattr(wrapper, "__script_if_tracing_wrapper") is True,
                )
                for wrapper in (raw_method, raw_static_method, raw_class_method)
            ),
            direct_bound_result[0] is instance,
            direct_bound_result[1],
            str(inspect.signature(direct_bound_wrapper)),
        )

    def invalid_target_outcome(self, decorator, target, call_args=()):
        wrapper = decorator(target)
        metadata = (
            wrapper.__name__,
            wrapper.__qualname__,
            wrapper.__module__.replace("torch_rs", "torch"),
            wrapper.__doc__,
            self.annotation_shape(wrapper),
            wrapper.__wrapped__ is target,
            getattr(wrapper, "__original_fn") is target,
            getattr(wrapper, "__script_if_tracing_wrapper") is True,
        )
        try:
            result = wrapper(*call_args)
        except Exception as error:
            outcome = ("error", self.normalize_error(error))
        else:
            outcome = ("result", result)
        return metadata, outcome

    def test_eager_calls_metadata_markers_and_binding_match_pytorch_2_13(self):
        self.assertEqual(self.eager_outcome(torch), self.eager_outcome(reference_torch))
        self.assertEqual(
            self.binding_outcome(torch),
            self.binding_outcome(reference_torch),
        )

    def test_signature_documentation_and_ownership_match(self):
        actual_jit = importlib.import_module("torch_rs.jit")
        expected_jit = importlib.import_module("torch.jit")
        actual_trace = importlib.import_module("torch_rs.jit._trace")
        expected_trace = importlib.import_module("torch.jit._trace")
        actual = actual_jit.script_if_tracing
        expected = expected_jit.script_if_tracing
        actual_helper = actual_trace._script_if_tracing
        expected_helper = expected_trace._script_if_tracing

        self.assertIs(torch.jit, actual_jit)
        self.assertIs(reference_torch.jit, expected_jit)
        self.assertIs(actual_jit._script_if_tracing, actual_helper)
        self.assertIs(expected_jit._script_if_tracing, expected_helper)
        self.assertIsNot(actual, actual_helper)
        self.assertIsNot(expected, expected_helper)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(str(inspect.signature(actual)), str(inspect.signature(expected)))
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(typing.get_type_hints(actual), typing.get_type_hints(expected))
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(actual.__module__.replace("torch_rs", "torch"), expected.__module__)
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
        self.assertEqual(
            self.annotation_shape(actual_helper),
            self.annotation_shape(expected_helper),
        )
        self.assertEqual(actual_helper.__name__, expected_helper.__name__)
        self.assertEqual(actual_helper.__qualname__, expected_helper.__qualname__)
        self.assertEqual(
            actual_helper.__module__.replace("torch_rs", "torch"),
            expected_helper.__module__,
        )
        self.assertIs(inspect.getmodule(actual_helper), actual_trace)
        self.assertIs(inspect.getmodule(expected_helper), expected_trace)

    def test_exports_copy_and_pickle_match_the_supported_scope(self):
        actual_jit = torch.jit
        expected_jit = reference_torch.jit
        actual_trace = actual_jit._trace
        expected_trace = expected_jit._trace
        actual = actual_jit.script_if_tracing
        expected = expected_jit.script_if_tracing
        wildcard_supported = {
            "annotate",
            "export",
            "ignore",
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
            [name for name in expected_jit.__all__ if name in wildcard_supported],
        )
        self.assertEqual(
            {name for name in vars(actual_jit) if not name.startswith("_")},
            public_supported,
        )
        self.assertIs(actual_jit._script_if_tracing, actual_trace._script_if_tracing)
        self.assertIs(
            expected_jit._script_if_tracing,
            expected_trace._script_if_tracing,
        )
        self.assertFalse(hasattr(actual_trace, "__all__"))
        self.assertFalse(hasattr(expected_trace, "__all__"))
        self.assertEqual(
            {name for name in vars(actual_trace) if not name.startswith("_")},
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

        actual_explicit = {}
        expected_explicit = {}
        exec("from torch_rs.jit import script_if_tracing", actual_explicit)
        exec("from torch.jit import script_if_tracing", expected_explicit)
        self.assertIs(actual_explicit["script_if_tracing"], actual)
        self.assertIs(expected_explicit["script_if_tracing"], expected)

        actual_helper_namespace = {}
        expected_helper_namespace = {}
        exec(
            "from torch_rs.jit._trace import _script_if_tracing",
            actual_helper_namespace,
        )
        exec(
            "from torch.jit._trace import _script_if_tracing",
            expected_helper_namespace,
        )
        self.assertIs(
            actual_helper_namespace["_script_if_tracing"],
            actual_trace._script_if_tracing,
        )
        self.assertIs(
            expected_helper_namespace["_script_if_tracing"],
            expected_trace._script_if_tracing,
        )

        self.assertEqual(
            torch.__all__.count("script_if_tracing"),
            reference_torch.__all__.count("script_if_tracing"),
        )
        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("script_if_tracing", namespace)
            self.assertFalse(hasattr(module, "script_if_tracing"))

        for actual_value, expected_value in (
            (actual, expected),
            (actual_trace._script_if_tracing, expected_trace._script_if_tracing),
        ):
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

        for value in (
            _actual_picklable_script_if_tracing_function,
            _expected_picklable_script_if_tracing_function,
        ):
            self.assertIs(copy.copy(value), value)
            self.assertIs(copy.deepcopy(value), value)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                self.assertIs(pickle.loads(pickle.dumps(value, protocol)), value)

    def test_invalid_calls_targets_and_forwarded_errors_match_pytorch_2_13(self):
        actual = torch.jit.script_if_tracing
        expected = reference_torch.jit.script_if_tracing
        decorator_calls = (
            (lambda function: function()),
            (lambda function: function(lambda: None, lambda: None)),
            (lambda function: function(target=lambda: None)),
            (lambda function: function(lambda: None, fn=lambda: None)),
        )
        for case, call in enumerate(decorator_calls):
            with self.subTest(decorator_case=case):
                self.assert_error_matches(
                    lambda: call(actual),
                    lambda: call(expected),
                )

        target_factories = (
            lambda: None,
            lambda: 1,
            lambda: "invalid",
            list,
            property,
            lambda: classmethod(lambda cls: cls),
        )
        for case, factory in enumerate(target_factories):
            with self.subTest(target_case=case):
                self.assertEqual(
                    self.invalid_target_outcome(actual, factory()),
                    self.invalid_target_outcome(expected, factory()),
                )

        def make_function():
            def function(value, /, offset, *, scale):
                return (value + offset) * scale

            return function

        actual_wrapper = actual(make_function())
        expected_wrapper = expected(make_function())
        wrapper_calls = (
            lambda function: function(),
            lambda function: function(1, 2, 3),
            lambda function: function(value=1, offset=2, scale=3),
            lambda function: function(1, 2, scale=3, unknown=4),
        )
        for case, call in enumerate(wrapper_calls):
            with self.subTest(wrapper_case=case):
                self.assert_error_matches(
                    lambda: call(actual_wrapper),
                    lambda: call(expected_wrapper),
                )

    def test_reference_only_trace_bounds_the_unsupported_compiled_path(self):
        actual_wrapper = torch.jit.script_if_tracing(lambda value: value + 1)
        self.assertEqual(actual_wrapper(2), 3)
        self.assertIs(torch.jit.is_tracing(), False)

        expected_wrapper = reference_torch.jit.script_if_tracing(
            _reference_control_flow
        )
        actual_trace_states = [torch.jit.is_tracing()]

        def trace_entry(value):
            actual_trace_states.append(torch.jit.is_tracing())
            return expected_wrapper(value)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            traced = reference_torch.jit.trace(
                trace_entry,
                reference_torch.tensor([1.0]),
                check_trace=False,
            )

        self.assertEqual(traced(reference_torch.tensor([2.0])).item(), 4.0)
        self.assertEqual(traced(reference_torch.tensor([-1.0])).item(), -3.0)
        actual_trace_states.append(torch.jit.is_tracing())
        self.assertEqual(actual_trace_states, [False, False, False])
        self.assertFalse(hasattr(torch.jit, "script"))
        self.assertFalse(hasattr(torch.jit, "trace"))
        self.assertTrue(callable(reference_torch.jit.script))
        self.assertTrue(callable(reference_torch.jit.trace))


if __name__ == "__main__":
    unittest.main()
