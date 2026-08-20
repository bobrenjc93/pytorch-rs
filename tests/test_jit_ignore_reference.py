import copy
import importlib
import inspect
import pickle
import pickletools
import types
import unittest
import warnings

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


def _actual_picklable_ignored_function(value):
    return value


torch.jit.ignore(_actual_picklable_ignored_function)


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class JitIgnoreReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "jit.ignore differentials require pinned PyTorch 2.13.0"
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
            shape.append((opcode.name, argument))
        return shape

    def warning_shape(self, caught):
        return tuple(
            (
                type(warning.message).__name__,
                str(warning.message),
                warning.message.args,
                warning.category.__name__,
                warning.filename,
                warning.lineno,
            )
            for warning in caught
        )

    def bare_outcome(self, module):
        sentinel = object()

        def function(value, *, option=sentinel):
            """function documentation"""
            return value, option

        function.custom_attribute = sentinel
        before = (
            function.__name__,
            function.__qualname__,
            function.__doc__,
            function.__annotations__.copy(),
            function.__defaults__,
            function.__kwdefaults__.copy(),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = module.jit.ignore(function, ignored_keyword="ignored")
        after = (
            function.__name__,
            function.__qualname__,
            function.__doc__,
            function.__annotations__,
            function.__defaults__,
            function.__kwdefaults__,
        )
        modifier = module._jit_internal.FunctionModifiers.IGNORE
        return (
            result is function,
            before == after,
            function.custom_attribute is sentinel,
            function("value") == ("value", sentinel),
            function._torchscript_modifier,
            function._torchscript_modifier is modifier,
            self.warning_shape(caught),
            copy.copy(function) is function,
            copy.deepcopy(function) is function,
        )

    def callable_outcome(self, module):
        class CallableTarget:
            def __call__(self, value):
                return value + 1

        target = CallableTarget()
        result = module.jit.ignore(target)
        modifier = module._jit_internal.FunctionModifiers.IGNORE
        return (
            result is target,
            target(3),
            target._torchscript_modifier,
            target._torchscript_modifier is modifier,
        )

    def factory_outcome(self, module, args, kwargs):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            decorator = module.jit.ignore(*args, **kwargs)

        decorator_metadata = (
            type(decorator) is types.FunctionType,
            str(inspect.signature(decorator)),
            decorator.__annotations__,
            decorator.__name__,
            decorator.__qualname__,
            decorator.__module__.replace("torch_rs", "torch"),
            decorator.__doc__,
            decorator.__defaults__,
            decorator.__kwdefaults__,
            decorator.__dict__,
            hasattr(decorator, "__text_signature__"),
            tuple(cell.cell_contents for cell in decorator.__closure__),
            copy.copy(decorator) is decorator,
            copy.deepcopy(decorator) is decorator,
        )
        pickle_errors = []
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            try:
                pickle.dumps(decorator, protocol=protocol)
            except Exception as error:
                def normalize(value):
                    if isinstance(value, str):
                        return value.replace(repr(decorator), "<decorator>")
                    return value

                pickle_errors.append(
                    (
                        type(error).__name__,
                        normalize(str(error)),
                        tuple(normalize(value) for value in error.args),
                    )
                )

        def function(value):
            return value

        function._torchscript_modifier = object()
        with warnings.catch_warnings(record=True) as apply_warnings:
            warnings.simplefilter("always")
            result = decorator(function)
        modifier = function._torchscript_modifier
        expected_modifier = (
            module._jit_internal.FunctionModifiers.UNUSED
            if modifier == module._jit_internal.FunctionModifiers.UNUSED
            else module._jit_internal.FunctionModifiers.IGNORE
        )
        return (
            self.warning_shape(caught),
            decorator_metadata,
            pickle_errors,
            self.warning_shape(apply_warnings),
            result is function,
            function("value"),
            modifier,
            modifier is expected_modifier,
        )

    def overwrite_outcome(self, module):
        def function(value):
            return value

        module.jit.export(function)
        export_state = function._torchscript_modifier
        module.jit.ignore(function)
        ignore_state = function._torchscript_modifier
        module.jit.unused(function)
        unused_state = function._torchscript_modifier
        module.jit.ignore()(function)
        return (
            export_state,
            ignore_state,
            unused_state,
            function._torchscript_modifier,
            function("eager"),
        )

    def test_bare_factory_callable_and_modifier_semantics_match(self):
        self.assertEqual(
            self.bare_outcome(torch), self.bare_outcome(reference_torch)
        )
        self.assertEqual(
            self.callable_outcome(torch), self.callable_outcome(reference_torch)
        )
        self.assertEqual(
            self.overwrite_outcome(torch), self.overwrite_outcome(reference_torch)
        )

        cases = (
            ((), {}),
            ((False,), {}),
            ((), {"drop": False}),
            ((), {"drop_on_export": False}),
            ((), {"unrecognized": "ignored"}),
            ((True,), {}),
            ((), {"drop": True}),
            ((True,), {"drop_on_export": False}),
            ((), {"drop_on_export": True}),
            ((False,), {"drop_on_export": True}),
            ((), {"drop_on_export": "truthy"}),
            ((), {"drop_on_export": True, "unrecognized": "ignored"}),
        )
        for args, kwargs in cases:
            with self.subTest(args=args, kwargs=kwargs):
                self.assertEqual(
                    self.factory_outcome(torch, args, kwargs),
                    self.factory_outcome(reference_torch, args, kwargs),
                )

    def test_function_and_factory_metadata_match_pytorch_2_13(self):
        actual_jit = importlib.import_module("torch_rs.jit")
        expected_jit = importlib.import_module("torch.jit")
        actual_internal = importlib.import_module("torch_rs._jit_internal")
        expected_internal = importlib.import_module("torch._jit_internal")
        actual = actual_jit.ignore
        expected = expected_jit.ignore

        self.assertIs(torch.jit, actual_jit)
        self.assertIs(reference_torch.jit, expected_jit)
        self.assertIs(torch._jit_internal, actual_internal)
        self.assertIs(reference_torch._jit_internal, expected_internal)
        self.assertIs(actual, actual_internal.ignore)
        self.assertIs(expected, expected_internal.ignore)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(str(inspect.signature(actual)), str(inspect.signature(expected)))
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertIs(inspect.getmodule(actual), actual_internal)
        self.assertIs(inspect.getmodule(expected), expected_internal)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )

        self.assertEqual(
            self.factory_outcome(torch, (), {}),
            self.factory_outcome(reference_torch, (), {}),
        )

    def test_exports_copying_and_pickling_match_the_supported_scope(self):
        actual_jit = torch.jit
        expected_jit = reference_torch.jit
        actual = actual_jit.ignore
        expected = expected_jit.ignore

        self.assertEqual(
            actual_jit.__all__,
            [
                name
                for name in expected_jit.__all__
                if name in {"annotate", "export", "ignore", "unused"}
            ],
        )
        self.assertEqual(
            torch.__all__.count("jit"), reference_torch.__all__.count("jit")
        )
        self.assertEqual(
            torch.__all__.count("ignore"),
            reference_torch.__all__.count("ignore"),
        )

        actual_namespace = {}
        expected_namespace = {}
        exec("from torch_rs.jit import *", actual_namespace)
        exec("from torch.jit import *", expected_namespace)
        self.assertEqual(
            {name for name in actual_namespace if not name.startswith("__")},
            {"annotate", "export", "ignore", "unused"},
        )
        self.assertIs(actual_namespace["ignore"], actual)
        self.assertIs(expected_namespace["ignore"], expected)

        for actual_value, expected_value in (
            (actual, expected),
            (
                torch._jit_internal.FunctionModifiers,
                reference_torch._jit_internal.FunctionModifiers,
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

        self.assertIs(
            _actual_picklable_ignored_function._torchscript_modifier,
            torch._jit_internal.FunctionModifiers.IGNORE,
        )
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            self.assertIs(
                pickle.loads(
                    pickle.dumps(_actual_picklable_ignored_function, protocol)
                ),
                _actual_picklable_ignored_function,
            )

    def test_errors_and_truthiness_failures_match_pytorch_2_13(self):
        actual = torch.jit.ignore
        expected = reference_torch.jit.ignore

        class Example:
            def method(self):
                return None

        class InvalidDrop:
            def __repr__(self):
                return "invalid-drop"

        class ExplodingTruth:
            def __bool__(self):
                raise ValueError("truthiness failed")

        class SlottedCallable:
            __slots__ = ()

            def __call__(self):
                return None

        invalid_drop = InvalidDrop()
        exploding_truth = ExplodingTruth()
        slotted_callable = SlottedCallable()
        cases = (
            lambda function: function(None),
            lambda function: function(1),
            lambda function: function(-1),
            lambda function: function("invalid"),
            lambda function: function(invalid_drop),
            lambda function: function(False, True),
            lambda function: function(False, drop=True),
            lambda function: function(len),
            lambda function: function(Example().method),
            lambda function: function(slotted_callable),
            lambda function: function()(None),
            lambda function: function()(1),
            lambda function: function()(len),
            lambda function: function()(Example().method),
            lambda function: function()(property()),
            lambda function: function()(property(1)),
            lambda function: function(drop_on_export=exploding_truth),
        )
        for case, call in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(
                    lambda: call(actual),
                    lambda: call(expected),
                )

    def test_supported_boundary_remains_eager_decorators_and_annotate_only(self):
        expected_public = {
            name for name in vars(reference_torch.jit) if not name.startswith("_")
        }
        self.assertEqual(
            {name for name in vars(torch.jit) if not name.startswith("_")},
            {"annotate", "export", "ignore", "unused"},
        )
        for name in ("script", "trace", "is_scripting", "is_tracing"):
            with self.subTest(name=name):
                self.assertIn(name, expected_public)
                self.assertFalse(hasattr(torch.jit, name))

        actual_value = object()
        expected_value = object()
        self.assertIs(torch.jit.annotate(int, actual_value), actual_value)
        self.assertIs(
            reference_torch.jit.annotate(int, expected_value), expected_value
        )
        self.assertTrue(hasattr(torch.jit, "export"))
        self.assertTrue(hasattr(torch.jit, "unused"))


if __name__ == "__main__":
    unittest.main()
