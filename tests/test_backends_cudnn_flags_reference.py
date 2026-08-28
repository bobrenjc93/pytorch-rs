import copy
import importlib
import inspect
import pickle
import pickletools
import re
import sys
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


DEFAULT_STATE = (True, False, 10, False, True)
TARGET_STATE = (False, True, 17, True, False)


class _ContextBodyError(Exception):
    pass


class _StringMode(str):
    def __eq__(self, other):
        raise AssertionError("mode validation must not dispatch equality")

    def __str__(self):
        raise AssertionError("mode validation must not dispatch string conversion")


class _BytesMode(bytes):
    def __eq__(self, other):
        raise AssertionError("mode validation must not dispatch equality")

    def decode(self, *args, **kwargs):
        raise AssertionError("mode validation must not dispatch byte decoding")


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CudnnFlagsReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "backends.cudnn.flags differentials require pinned "
                "PyTorch 2.13.0"
            )
        if not reference_torch.backends.cudnn.is_available():
            raise unittest.SkipTest(
                "cudnn.flags benchmark-limit differentials require a "
                "cuDNN-built reference PyTorch"
            )

    def fresh_cudnn_module(self, root):
        module_name = f"{root.__name__}.backends.cudnn"
        sys.modules.pop(module_name, None)
        if hasattr(root.backends, "cudnn"):
            del root.backends.cudnn
        module = importlib.import_module(module_name)
        root.backends.cudnn = module
        return module

    def setUp(self):
        self.actual = self.fresh_cudnn_module(torch)
        self.expected = self.fresh_cudnn_module(reference_torch)
        self.original_actual = self.states(self.actual)
        self.original_expected = self.states(self.expected)
        self.set_states(self.actual, DEFAULT_STATE)
        self.set_states(self.expected, DEFAULT_STATE)

    def tearDown(self):
        actual = self.fresh_cudnn_module(torch)
        expected = self.fresh_cudnn_module(reference_torch)
        self.set_states(actual, self.original_actual)
        self.set_states(expected, self.original_expected)

    def states(self, module):
        return (
            module.enabled,
            module.benchmark,
            module.benchmark_limit,
            module.deterministic,
            module.allow_tf32,
        )

    def set_states(self, module, states):
        (
            module.enabled,
            module.benchmark,
            module.benchmark_limit,
            module.deterministic,
            module.allow_tf32,
        ) = states

    def normalize(self, value):
        return re.sub(
            r"0x[0-9a-fA-F]+",
            "0x...",
            str(value).replace("torch_rs", "torch"),
        )

    def capture_error(self, call):
        try:
            call()
        except Exception as error:
            return (
                type(error).__name__,
                self.normalize(error),
                tuple(self.normalize(argument) for argument in error.args),
            )
        self.fail("expected the call to fail")

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

    def behavior_contract(self, module):
        outcomes = []
        for initial in (DEFAULT_STATE, (False, True, -9, True, False)):
            self.set_states(module, initial)
            context = module.flags()
            before_entry = self.states(module)
            entered = context.__enter__()
            inside = self.states(module)
            exit_result = context.__exit__(None, None, None)
            outcomes.append(
                (
                    before_entry,
                    entered,
                    inside,
                    exit_result,
                    self.states(module),
                )
            )

            self.set_states(module, initial)
            context = module.flags(*TARGET_STATE)
            before_entry = self.states(module)
            entered = context.__enter__()
            inside = self.states(module)
            exit_result = context.__exit__(None, None, None)
            outcomes.append(
                (
                    before_entry,
                    entered,
                    inside,
                    exit_result,
                    self.states(module),
                )
            )

            self.set_states(module, initial)
            with module.flags(None, None, None, None, None) as entered:
                inside = self.states(module)
            outcomes.append((entered, inside, self.states(module)))

        self.set_states(module, DEFAULT_STATE)
        outer_state = (False, True, 21, True, False)
        inner_state = (True, False, -7, False, True)
        with module.flags(*outer_state) as outer:
            outer_inside = self.states(module)
            with module.flags(*inner_state) as inner:
                inner_inside = self.states(module)
            after_inner = self.states(module)
        outcomes.append(
            (
                outer,
                outer_inside,
                inner,
                inner_inside,
                after_inner,
                self.states(module),
            )
        )

        marker = _ContextBodyError("body failed")
        try:
            with module.flags(*TARGET_STATE) as entered:
                before_error = self.states(module)
                raise marker
        except Exception as error:
            outcomes.append(
                (
                    entered,
                    before_error,
                    error is marker,
                    type(error).__name__,
                    error.args,
                    self.states(module),
                )
            )
        else:
            self.fail("the context suppressed its body exception")

        observations = []

        @module.flags(*TARGET_STATE)
        def decorated(value):
            observations.append(self.states(module))
            if value == "raise":
                raise _ContextBodyError("decorated body failed")
            return value

        first = decorated("first")
        after_first = self.states(module)
        second = decorated("second")
        after_second = self.states(module)
        try:
            decorated("raise")
        except Exception as error:
            decorator_error = (type(error).__name__, error.args)
        else:
            self.fail("the decorated function suppressed its exception")
        outcomes.append(
            (
                first,
                after_first,
                second,
                after_second,
                decorator_error,
                observations,
                self.states(module),
            )
        )
        return outcomes

    def failed_entry_contract(self, module):
        cases = (
            (1, True, 17, True, False),
            (False, 1, 17, True, False),
            (False, True, "invalid", True, False),
            (False, True, 17, 1, False),
            (False, True, 17, True, 1),
        )
        outcomes = []
        for arguments in cases:
            self.set_states(module, DEFAULT_STATE)
            context = module.flags(*arguments)
            before_entry = self.states(module)
            error = self.capture_error(context.__enter__)
            outcomes.append((before_entry, error, self.states(module)))

        self.set_states(module, DEFAULT_STATE)
        context = module.flags(
            False,
            1,
            17,
            True,
            False,
            fp32_precision="ieee",
            depthwise_kernel="cudnn",
        )
        outcomes.append(
            (
                self.capture_error(context.__enter__),
                self.states(module),
            )
        )
        return outcomes

    def test_supported_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.behavior_contract(self.actual),
            self.behavior_contract(self.expected),
        )

    def test_failed_entry_order_and_partial_mutation_match_pytorch_2_13(self):
        self.assertEqual(
            self.failed_entry_contract(self.actual),
            self.failed_entry_contract(self.expected),
        )

    def test_argument_binding_errors_match_pytorch_2_13(self):
        actual_calls = (
            lambda: self.actual.flags(
                False,
                False,
                10,
                False,
                True,
                "none",
                "auto",
                None,
            ),
            lambda: self.actual.flags(_enabled=False),
            lambda: self.actual.flags(False, enabled=True),
        )
        expected_calls = (
            lambda: self.expected.flags(
                False,
                False,
                10,
                False,
                True,
                "none",
                "auto",
                None,
            ),
            lambda: self.expected.flags(_enabled=False),
            lambda: self.expected.flags(False, enabled=True),
        )
        for actual_call, expected_call in zip(actual_calls, expected_calls):
            with self.subTest(call=actual_call):
                self.set_states(self.actual, DEFAULT_STATE)
                self.set_states(self.expected, DEFAULT_STATE)
                self.assertEqual(
                    self.capture_error(actual_call),
                    self.capture_error(expected_call),
                )
                self.assertEqual(self.states(self.actual), DEFAULT_STATE)
                self.assertEqual(self.states(self.expected), DEFAULT_STATE)

    def test_callable_metadata_imports_copying_and_pickling_match(self):
        actual = self.actual
        expected = self.expected
        actual_function = actual.flags
        expected_function = expected.flags
        actual_wrapped = actual_function.__wrapped__
        expected_wrapped = expected_function.__wrapped__

        self.assertEqual(type(actual).__name__, type(expected).__name__)
        self.assertEqual(
            type(actual).__module__.replace("torch_rs", "torch"),
            type(expected).__module__,
        )
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(
            hasattr(actual, "__all__"),
            hasattr(expected, "__all__"),
        )
        self.assertEqual(
            {name for name in vars(actual) if not name.startswith("_")},
            {name for name in vars(expected) if not name.startswith("_")},
        )

        for actual_value, expected_value in (
            (actual_function, expected_function),
            (actual_wrapped, expected_wrapped),
        ):
            with self.subTest(wrapped=actual_value is actual_wrapped):
                self.assertIs(type(actual_value), types.FunctionType)
                self.assertIs(type(expected_value), types.FunctionType)
                self.assertEqual(
                    str(inspect.signature(actual_value)),
                    str(inspect.signature(expected_value)),
                )
                self.assertEqual(
                    inspect.get_annotations(actual_value),
                    inspect.get_annotations(expected_value),
                )
                self.assertEqual(actual_value.__name__, expected_value.__name__)
                self.assertEqual(
                    actual_value.__qualname__,
                    expected_value.__qualname__,
                )
                self.assertEqual(
                    actual_value.__module__.replace("torch_rs", "torch"),
                    expected_value.__module__,
                )
                self.assertEqual(actual_value.__doc__, expected_value.__doc__)
                self.assertEqual(
                    actual_value.__defaults__,
                    expected_value.__defaults__,
                )
                self.assertEqual(
                    actual_value.__kwdefaults__,
                    expected_value.__kwdefaults__,
                )
                self.assertEqual(
                    set(actual_value.__dict__),
                    set(expected_value.__dict__),
                )
                self.assertEqual(
                    hasattr(actual_value, "__text_signature__"),
                    hasattr(expected_value, "__text_signature__"),
                )
                self.assertEqual(
                    actual_value.__code__.co_freevars,
                    expected_value.__code__.co_freevars,
                )
                self.assertEqual(
                    actual_value.__code__.co_cellvars,
                    expected_value.__code__.co_cellvars,
                )

        actual_direct = {}
        expected_direct = {}
        actual_wildcard = {}
        expected_wildcard = {}
        exec("from torch_rs.backends.cudnn import flags", actual_direct)
        exec("from torch.backends.cudnn import flags", expected_direct)
        exec("from torch_rs.backends.cudnn import *", actual_wildcard)
        exec("from torch.backends.cudnn import *", expected_wildcard)
        self.assertIs(actual_direct["flags"], actual_function)
        self.assertIs(expected_direct["flags"], expected_function)
        self.assertEqual(
            {name for name in actual_wildcard if not name.startswith("__")},
            {name for name in expected_wildcard if not name.startswith("__")},
        )
        self.assertFalse(hasattr(actual, "set_flags"))
        self.assertTrue(hasattr(expected, "set_flags"))

        self.assertIs(copy.copy(actual_function), actual_function)
        self.assertIs(copy.deepcopy(actual_function), actual_function)
        self.assertIs(copy.copy(expected_function), expected_function)
        self.assertIs(copy.deepcopy(expected_function), expected_function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertEqual(
                    self.pickle_shape(actual_function, protocol),
                    self.pickle_shape(expected_function, protocol),
                )

        actual_context = actual_function(*TARGET_STATE)
        expected_context = expected_function(*TARGET_STATE)
        self.assertEqual(type(actual_context), type(expected_context))
        self.assertEqual(actual_context.__doc__, expected_context.__doc__)
        self.assertIs(actual_context.func, actual_wrapped)
        self.assertIs(expected_context.func, expected_wrapped)
        self.assertEqual(actual_context.args, expected_context.args)
        self.assertEqual(actual_context.kwds, expected_context.kwds)
        actual_copy = copy.copy(actual_context)
        expected_copy = copy.copy(expected_context)
        self.assertIs(actual_copy.gen, actual_context.gen)
        self.assertIs(expected_copy.gen, expected_context.gen)
        for context in (actual_context, expected_context):
            with self.assertRaisesRegex(
                TypeError,
                "cannot pickle 'generator' object",
            ):
                copy.deepcopy(context)

    def test_unsupported_mode_boundary_is_explicit(self):
        for fp32_precision, depthwise_kernel in (
            ("none", "auto"),
            (b"none", b"auto"),
            (_StringMode("none"), _StringMode("auto")),
            (_BytesMode(b"none"), _BytesMode(b"auto")),
            (None, None),
        ):
            with self.subTest(
                supported_fp32=type(fp32_precision).__name__,
                supported_depthwise=type(depthwise_kernel).__name__,
            ):
                for module in (self.actual, self.expected):
                    self.set_states(module, DEFAULT_STATE)
                    with module.flags(
                        *TARGET_STATE,
                        fp32_precision=fp32_precision,
                        depthwise_kernel=depthwise_kernel,
                    ) as entered:
                        self.assertIsNone(entered)
                        self.assertEqual(self.states(module), TARGET_STATE)
                    self.assertEqual(self.states(module), DEFAULT_STATE)

        for keywords, message in (
            (
                {"fp32_precision": "ieee"},
                "torch.backends.cudnn.flags() only supports "
                "fp32_precision='none'",
            ),
            (
                {"depthwise_kernel": "cudnn"},
                "torch.backends.cudnn.flags() only supports "
                "depthwise_kernel='auto'",
            ),
        ):
            with self.subTest(keywords=keywords):
                self.set_states(self.actual, DEFAULT_STATE)
                context = self.actual.flags(
                    None,
                    None,
                    None,
                    None,
                    None,
                    **keywords,
                )
                with self.assertRaises(NotImplementedError) as raised:
                    context.__enter__()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(self.states(self.actual), DEFAULT_STATE)

        with self.expected.flags(
            None,
            None,
            None,
            None,
            None,
            fp32_precision="ieee",
        ):
            self.assertEqual(self.expected.fp32_precision, "ieee")
        with self.expected.flags(
            None,
            None,
            None,
            None,
            None,
            depthwise_kernel="cudnn",
        ):
            self.assertEqual(self.expected.depthwise_kernel, "cudnn")


if __name__ == "__main__":
    unittest.main()
