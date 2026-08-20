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


def _actual_picklable_ignore_function(value):
    return value


def _expected_picklable_ignore_function(value):
    return value


torch.jit.ignore(_actual_picklable_ignore_function)
if reference_torch is not None:
    reference_torch.jit.ignore(_expected_picklable_ignore_function)


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class JitIgnoreReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "jit.ignore differentials require pinned PyTorch 2.13.0"
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

    def bare_outcome(self, module):
        sentinel = object()

        def function(value, *, option=sentinel):
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
        result = module.jit.ignore(function)
        modifier = module._jit_internal.FunctionModifiers.IGNORE
        after = (
            function.__name__,
            function.__qualname__,
            function.__doc__,
            function.__annotations__,
            function.__defaults__,
            function.__kwdefaults__,
        )
        return (
            result is function,
            function("value") == ("value", sentinel),
            function.custom_attribute is sentinel,
            before == after,
            function._torchscript_modifier,
            function._torchscript_modifier is modifier,
            copy.copy(function) is function,
            copy.deepcopy(function) is function,
        )

    def factory_outcome(self, module, make_factory):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            decorator = make_factory(module)

        def function(value):
            return value + 1

        result = decorator(function)
        modifier = module._jit_internal.FunctionModifiers
        return (
            result is function,
            function(4),
            function._torchscript_modifier,
            function._torchscript_modifier is modifier.IGNORE,
            function._torchscript_modifier is modifier.UNUSED,
            tuple((item.category.__name__, str(item.message)) for item in caught),
            str(inspect.signature(decorator)),
            decorator.__name__,
            decorator.__qualname__,
            decorator.__module__.replace("torch_rs", "torch"),
            decorator.__annotations__,
            decorator.__defaults__,
            decorator.__kwdefaults__,
            decorator.__dict__,
            copy.copy(decorator) is decorator,
            copy.deepcopy(decorator) is decorator,
        )

    def test_bare_factory_drop_and_legacy_semantics_match(self):
        self.assertEqual(self.bare_outcome(torch), self.bare_outcome(reference_torch))

        factories = (
            lambda module: module.jit.ignore(),
            lambda module: module.jit.ignore(False),
            lambda module: module.jit.ignore(drop=False),
            lambda module: module.jit.ignore(drop_on_export=False),
            lambda module: module.jit.ignore(unrecognized="ignored"),
            lambda module: module.jit.ignore(True),
            lambda module: module.jit.ignore(drop=True),
            lambda module: module.jit.ignore(drop_on_export=True),
            lambda module: module.jit.ignore(False, drop_on_export=True),
            lambda module: module.jit.ignore(True, drop_on_export=True),
            lambda module: module.jit.ignore(True, drop_on_export=False),
            lambda module: module.jit.ignore(
                drop_on_export="truthy", unrecognized="ignored"
            ),
        )
        for make_factory in factories:
            with self.subTest(make_factory=make_factory):
                self.assertEqual(
                    self.factory_outcome(torch, make_factory),
                    self.factory_outcome(reference_torch, make_factory),
                )

    def test_methods_callable_objects_and_modifier_overwrites_match(self):
        def outcome(module):
            class Example:
                @module.jit.ignore
                def bare(self, value):
                    return value + 1

                @module.jit.ignore()
                def factory(self, value):
                    return value + 2

            class CallableTarget:
                def __call__(self, value):
                    return value * 2

            target = CallableTarget()
            result = module.jit.ignore(target)

            def function():
                return "eager"

            module.jit.export(function)
            export_state = function._torchscript_modifier
            module.jit.ignore(function)
            ignore_state = function._torchscript_modifier
            module.jit.unused(function)
            unused_state = function._torchscript_modifier
            return (
                Example().bare(3),
                Example().factory(3),
                Example.__dict__["bare"]._torchscript_modifier,
                Example.__dict__["factory"]._torchscript_modifier,
                result is target,
                target(5),
                target._torchscript_modifier,
                export_state,
                ignore_state,
                unused_state,
                function(),
            )

        self.assertEqual(outcome(torch), outcome(reference_torch))

    def test_signature_documentation_and_ownership_match(self):
        actual_jit = importlib.import_module("torch_rs.jit")
        expected_jit = importlib.import_module("torch.jit")
        actual_internal = importlib.import_module("torch_rs._jit_internal")
        expected_internal = importlib.import_module("torch._jit_internal")
        actual = actual_jit.ignore
        expected = expected_jit.ignore

        self.assertIs(torch.jit, actual_jit)
        self.assertIs(reference_torch.jit, expected_jit)
        self.assertIs(actual, actual_internal.ignore)
        self.assertIs(expected, expected_internal.ignore)
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

    def test_exports_copy_and_pickle_match_supported_scope(self):
        actual = torch.jit.ignore
        expected = reference_torch.jit.ignore
        wildcard_supported = {"annotate", "export", "ignore", "unused"}
        self.assertEqual(
            torch.jit.__all__,
            [
                name
                for name in reference_torch.jit.__all__
                if name in wildcard_supported
            ],
        )
        self.assertEqual(
            {name for name in vars(torch.jit) if not name.startswith("_")},
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

        for value in (
            _actual_picklable_ignore_function,
            _expected_picklable_ignore_function,
        ):
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                self.assertIs(pickle.loads(pickle.dumps(value, protocol)), value)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            actual_factory = actual()
            expected_factory = expected()
            self.assert_error_matches(
                lambda: pickle.dumps(actual_factory, protocol),
                lambda: pickle.dumps(expected_factory, protocol),
            )

    def test_call_and_invalid_target_errors_match_pytorch_2_13(self):
        actual = torch.jit.ignore
        expected = reference_torch.jit.ignore

        class Example:
            def method(self):
                return None

        cases = (
            lambda function: function(False, True),
            lambda function: function(False, drop=True),
            lambda function: function(None),
            lambda function: function(1),
            lambda function: function(1.5),
            lambda function: function("invalid"),
            lambda function: function([]),
            lambda function: function(len),
            lambda function: function(Example().method),
            lambda function: function(property()),
            lambda function: function()(None),
            lambda function: function()(1),
            lambda function: function()(len),
            lambda function: function()(Example().method),
            lambda function: function()(property()),
        )
        for case, call in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(
                    lambda: call(actual),
                    lambda: call(expected),
                )

    def test_supported_boundary_remains_eager_only(self):
        expected_public = {
            name for name in vars(reference_torch.jit) if not name.startswith("_")
        }
        for name in ("script", "trace"):
            with self.subTest(name=name):
                self.assertIn(name, expected_public)
                self.assertFalse(hasattr(torch.jit, name))

        self.assertIs(torch.jit.is_scripting(), False)

        self.assertTrue(hasattr(reference_torch, "compile"))
        self.assertFalse(hasattr(torch, "compile"))

        actual_value = object()
        expected_value = object()
        self.assertIs(torch.jit.annotate(int, actual_value), actual_value)
        self.assertIs(
            reference_torch.jit.annotate(int, expected_value), expected_value
        )


if __name__ == "__main__":
    unittest.main()
