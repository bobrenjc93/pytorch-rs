import collections.abc
import copy
import importlib
import inspect
import pickle
import pickletools
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


def _expected_picklable_script_if_tracing_function(value):
    return value


if reference_torch is not None:
    _expected_picklable_script_if_tracing_function = (
        reference_torch.jit.script_if_tracing(
            _expected_picklable_script_if_tracing_function
        )
    )


def _reference_control_flow(value):
    if value > 0:
        return value + 1
    return value - 1


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class JitScriptIfTracingReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "jit.script_if_tracing differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

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

    def annotation_shape(self, function):
        shape = {}
        for name, annotation in function.__annotations__.items():
            parameters, result = typing.get_args(annotation)
            shape[name] = (
                typing.get_origin(annotation) is collections.abc.Callable,
                type(parameters).__name__,
                parameters.__name__,
                parameters.__module__.replace("torch_rs", "torch"),
                type(result).__name__,
                result.__name__,
                result.__module__.replace("torch_rs", "torch"),
                result.__covariant__,
            )
        return shape

    def eager_outcome(self, module):
        calls = []
        result = object()

        def function(positional, /, optional=None, *args, keyword, **kwargs):
            calls.append(
                (
                    positional,
                    optional,
                    args,
                    keyword,
                    kwargs,
                    module.jit.is_tracing(),
                )
            )
            return result

        wrapped = module.jit.script_if_tracing(function)
        returned = wrapped("first", "second", "third", keyword=4, named=5)

        error = RuntimeError("forwarded failure")

        def raising(value):
            raise error

        wrapped_raising = module.jit.script_if_tracing(raising)
        try:
            wrapped_raising("value")
        except RuntimeError as raised:
            forwarded_error = raised is error
        else:
            forwarded_error = False

        return (
            wrapped is not function,
            returned is result,
            calls,
            forwarded_error,
            wrapped.__wrapped__ is function,
            getattr(wrapped, "__original_fn") is function,
            getattr(wrapped, "__script_if_tracing_wrapper") is True,
        )

    def metadata_outcome(self, module):
        sentinel = object()

        def function(
            value: int,
            /,
            scale: float = 2.0,
            *,
            label: str = "value",
        ) -> tuple[int, float, str]:
            """Function documentation."""
            return value, scale, label

        function.custom_attribute = sentinel
        setattr(function, "__original_fn", "stale original")
        setattr(function, "__script_if_tracing_wrapper", False)
        wrapped = module.jit.script_if_tracing(function)
        return (
            type(wrapped).__name__,
            wrapped is not function,
            wrapped.__name__,
            wrapped.__qualname__,
            wrapped.__module__,
            wrapped.__doc__,
            wrapped.__annotations__,
            typing.get_type_hints(wrapped),
            wrapped.__defaults__,
            wrapped.__kwdefaults__,
            wrapped.custom_attribute is sentinel,
            wrapped.__wrapped__ is function,
            getattr(wrapped, "__original_fn") is function,
            getattr(wrapped, "__script_if_tracing_wrapper") is True,
            inspect.unwrap(wrapped) is function,
            str(inspect.signature(wrapped)),
            sorted(wrapped.__dict__),
            wrapped(3, label="scaled"),
        )

    def binding_outcome(self, module):
        class Example:
            @module.jit.script_if_tracing
            def method(self, value):
                return self is instance, value + 1

            @staticmethod
            @module.jit.script_if_tracing
            def static_method(value):
                return value + 2

            @classmethod
            @module.jit.script_if_tracing
            def class_method(cls, value):
                return cls is Example, value + 3

        instance = Example()

        class CallableTarget:
            def __call__(self, value, *, increment=1):
                return value + increment

        target = CallableTarget()
        wrapped_target = module.jit.script_if_tracing(target)
        raw_method = Example.__dict__["method"]
        return (
            instance.method(4),
            Example.static_method(4),
            instance.static_method(4),
            Example.class_method(4),
            instance.class_method(4),
            getattr(raw_method, "__original_fn").__qualname__
            == raw_method.__qualname__,
            getattr(raw_method, "__script_if_tracing_wrapper") is True,
            getattr(wrapped_target, "__original_fn") is target,
            str(inspect.signature(wrapped_target)),
            wrapped_target(4, increment=5),
            module.jit.script_if_tracing(len)([1, 2, 3]),
        )

    def test_eager_forwarding_metadata_and_binding_match_pytorch_2_13(self):
        self.assertEqual(
            self.eager_outcome(torch),
            self.eager_outcome(reference_torch),
        )
        self.assertEqual(
            self.metadata_outcome(torch),
            self.metadata_outcome(reference_torch),
        )
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
        actual_internal = actual_trace._script_if_tracing
        expected_internal = expected_trace._script_if_tracing

        self.assertIs(torch.jit, actual_jit)
        self.assertIs(reference_torch.jit, expected_jit)
        self.assertIs(actual_jit._trace, actual_trace)
        self.assertIs(expected_jit._trace, expected_trace)
        self.assertIsNot(actual, actual_internal)
        self.assertIsNot(expected, expected_internal)
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

        self.assertIs(type(actual_internal), types.FunctionType)
        self.assertIs(type(expected_internal), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual_internal)),
            str(inspect.signature(expected_internal)),
        )
        self.assertEqual(
            self.annotation_shape(actual_internal),
            self.annotation_shape(expected_internal),
        )
        self.assertEqual(actual_internal.__name__, expected_internal.__name__)
        self.assertEqual(actual_internal.__qualname__, expected_internal.__qualname__)
        self.assertEqual(
            actual_internal.__module__.replace("torch_rs", "torch"),
            expected_internal.__module__,
        )
        self.assertIs(inspect.getmodule(actual_internal), actual_trace)
        self.assertIs(inspect.getmodule(expected_internal), expected_trace)
        self.assertEqual(actual_internal.__doc__, expected_internal.__doc__)
        self.assertEqual(actual_internal.__defaults__, expected_internal.__defaults__)
        self.assertEqual(
            actual_internal.__kwdefaults__, expected_internal.__kwdefaults__
        )
        self.assertEqual(actual_internal.__dict__, expected_internal.__dict__)

    def test_exports_copy_and_pickle_match_the_supported_scope(self):
        actual_jit = torch.jit
        expected_jit = reference_torch.jit
        actual_trace = actual_jit._trace
        expected_trace = expected_jit._trace
        wildcard_supported = {
            "annotate",
            "export",
            "ignore",
            "isinstance",
            "script_if_tracing",
            "unused",
        }
        public_supported = {*wildcard_supported, "is_scripting", "is_tracing"}

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
            public_supported,
        )
        self.assertFalse(hasattr(actual_trace, "__all__"))
        self.assertFalse(hasattr(expected_trace, "__all__"))
        self.assertEqual(
            {name for name in vars(actual_trace) if not name.startswith("_")},
            {"is_tracing"},
        )
        self.assertIn("_script_if_tracing", vars(expected_trace))
        self.assertEqual(
            torch.__all__.count("script_if_tracing"),
            reference_torch.__all__.count("script_if_tracing"),
        )

        actual_namespace = {}
        expected_namespace = {}
        exec("from torch_rs.jit import *", actual_namespace)
        exec("from torch.jit import *", expected_namespace)
        self.assertEqual(
            {name for name in actual_namespace if not name.startswith("__")},
            wildcard_supported,
        )
        self.assertIs(
            actual_namespace["script_if_tracing"], actual_jit.script_if_tracing
        )
        self.assertIs(
            expected_namespace["script_if_tracing"], expected_jit.script_if_tracing
        )

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("script_if_tracing", namespace)
            self.assertFalse(hasattr(module, "script_if_tracing"))

        for actual_value, expected_value in (
            (actual_jit.script_if_tracing, expected_jit.script_if_tracing),
            (actual_trace._script_if_tracing, expected_trace._script_if_tracing),
            (
                _actual_picklable_script_if_tracing_function,
                _expected_picklable_script_if_tracing_function,
            ),
        ):
            self.assertIs(copy.copy(actual_value), actual_value)
            self.assertIs(copy.copy(expected_value), expected_value)
            self.assertIs(copy.deepcopy(actual_value), actual_value)
            self.assertIs(copy.deepcopy(expected_value), expected_value)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(value=actual_value, protocol=protocol):
                    self.assertEqual(
                        self.pickle_shape(actual_value, protocol),
                        self.pickle_shape(expected_value, protocol),
                    )
                    self.assertIs(
                        pickle.loads(pickle.dumps(actual_value, protocol)),
                        actual_value,
                    )
                    self.assertIs(
                        pickle.loads(pickle.dumps(expected_value, protocol)),
                        expected_value,
                    )

    def test_call_and_invalid_target_errors_match_pytorch_2_13(self):
        actual = torch.jit.script_if_tracing
        expected = reference_torch.jit.script_if_tracing
        cases = (
            lambda function: function(),
            lambda function: function(lambda: None, lambda: None),
            lambda function: function(function=lambda: None),
            lambda function: function(lambda: None, fn=lambda: None),
            lambda function: function(None)(),
            lambda function: function(1)(),
            lambda function: function(property())(),
            lambda function: function(len)(),
            lambda function: function(len)([], []),
        )
        for case, call in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(
                    lambda: call(actual),
                    lambda: call(expected),
                )

    def test_reference_only_tracing_bounds_the_unsupported_compiled_path(self):
        actual_states = []

        def actual_probe(value):
            actual_states.append(torch.jit.is_tracing())
            return value + 1

        actual_wrapped = torch.jit.script_if_tracing(actual_probe)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            actual_trace = reference_torch.jit.trace(
                actual_wrapped,
                reference_torch.tensor(1.0),
                check_trace=False,
            )
        self.assertEqual(actual_trace(reference_torch.tensor(2.0)).item(), 3.0)
        self.assertTrue(actual_states)
        self.assertTrue(all(state is False for state in actual_states))
        self.assertIs(torch.jit.is_tracing(), False)

        expected_wrapped = reference_torch.jit.script_if_tracing(
            _reference_control_flow
        )
        self.assertIs(reference_torch.jit.is_tracing(), False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            expected_trace = reference_torch.jit.trace(
                expected_wrapped,
                reference_torch.tensor(1.0),
                check_trace=False,
            )
        self.assertEqual(expected_trace(reference_torch.tensor(2.0)).item(), 3.0)
        self.assertEqual(expected_trace(reference_torch.tensor(-2.0)).item(), -3.0)
        self.assertIs(reference_torch.jit.is_tracing(), False)
        self.assertIs(torch.jit.is_tracing(), False)

    def test_script_and_trace_remain_outside_the_supported_boundary(self):
        self.assertTrue(callable(torch.jit.script_if_tracing))
        self.assertTrue(callable(reference_torch.jit.script_if_tracing))
        self.assertTrue(callable(reference_torch.jit.script))
        self.assertTrue(callable(reference_torch.jit.trace))
        for name in ("script", "trace"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.jit, name))
        self.assertIs(torch.jit.is_tracing(), False)
        self.assertFalse(hasattr(torch, "compile"))


if __name__ == "__main__":
    unittest.main()
