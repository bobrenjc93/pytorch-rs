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
DEFAULT_MODES = ("none", "auto")


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


class _BytearrayMode(bytearray):
    def __eq__(self, other):
        raise AssertionError("mode validation must not dispatch equality")

    def decode(self, *args, **kwargs):
        raise AssertionError("mode validation must not dispatch byte decoding")


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CudnnSetFlagsReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "backends.cudnn.set_flags differentials require pinned "
                "PyTorch 2.13.0"
            )
        if not reference_torch.backends.cudnn.is_available():
            raise unittest.SkipTest(
                "cudnn.set_flags benchmark-limit differentials require a "
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
        self.original_actual = self.actual.set_flags()
        self.original_expected = self.expected.set_flags()
        self.actual.set_flags(*DEFAULT_STATE, *DEFAULT_MODES)
        self.expected.set_flags(*DEFAULT_STATE, *DEFAULT_MODES)

    def tearDown(self):
        actual = self.fresh_cudnn_module(torch)
        expected = self.fresh_cudnn_module(reference_torch)
        actual.set_flags(*self.original_actual)
        expected.set_flags(*self.original_expected)

    def states(self, module):
        return (
            module.enabled,
            module.benchmark,
            module.benchmark_limit,
            module.deterministic,
            module.allow_tf32,
        )

    def set_default(self, module):
        module.set_flags(*DEFAULT_STATE, *DEFAULT_MODES)

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

    def transition_contract(self, module):
        outcomes = []
        self.set_default(module)
        for arguments in (
            (),
            TARGET_STATE,
            (None, None, None, None, None, None, None),
            (True, None, 10, False, True, "none", "auto"),
        ):
            result = module.set_flags(*arguments)
            outcomes.append(
                (
                    type(result) is tuple,
                    len(result),
                    result,
                    self.states(module),
                )
            )
        return outcomes

    def failed_update_contract(self, module):
        outcomes = []
        for arguments in (
            (1, True, 17, True, False),
            (False, 1, 17, True, False),
            (False, True, "invalid", True, False),
            (False, True, 17, 1, False),
            (False, True, 17, True, 1),
        ):
            self.set_default(module)
            outcomes.append(
                (
                    self.capture_error(lambda: module.set_flags(*arguments)),
                    self.states(module),
                )
            )
        return outcomes

    def reload_contract(self, root, module):
        old_setter = module.set_flags
        namespace = module.__dict__
        old_setter(*TARGET_STATE)
        reloaded = importlib.reload(module)
        preserved_state = self.states(module)
        new_result = module.set_flags(*DEFAULT_STATE)
        old_result = old_setter(*TARGET_STATE)
        final_result = module.set_flags(*DEFAULT_STATE)

        try:
            pickle.dumps(old_setter)
        except Exception as error:
            stale_error = (type(error).__name__, self.normalize(error))
        else:
            self.fail("a stale cuDNN setter remained pickleable")

        return (
            reloaded is not module,
            module.__dict__ is namespace,
            root.backends.cudnn is module,
            sys.modules[module.__name__] is reloaded,
            reloaded.m is module,
            module.set_flags is not old_setter,
            reloaded.set_flags is module.set_flags,
            preserved_state,
            new_result,
            old_result,
            final_result,
            stale_error,
        )

    def test_state_transitions_match_pytorch_2_13(self):
        self.assertEqual(
            self.transition_contract(self.actual),
            self.transition_contract(self.expected),
        )

    def test_failed_update_order_matches_pytorch_2_13(self):
        self.assertEqual(
            self.failed_update_contract(self.actual),
            self.failed_update_contract(self.expected),
        )

    def test_supported_trailing_modes_match_pytorch_2_13(self):
        for fp32_precision, depthwise_kernel in (
            ("none", "auto"),
            (b"none", b"auto"),
            (_StringMode("none"), _StringMode("auto")),
            (_BytesMode(b"none"), _BytesMode(b"auto")),
            (bytearray(b"none"), None),
            (_BytearrayMode(b"none"), None),
            (None, None),
        ):
            with self.subTest(
                fp32_precision=type(fp32_precision).__name__,
                depthwise_kernel=type(depthwise_kernel).__name__,
            ):
                self.set_default(self.actual)
                self.set_default(self.expected)
                actual_result = self.actual.set_flags(
                    *TARGET_STATE,
                    fp32_precision,
                    depthwise_kernel,
                )
                expected_result = self.expected.set_flags(
                    *TARGET_STATE,
                    fp32_precision,
                    depthwise_kernel,
                )
                self.assertEqual(actual_result, expected_result)
                self.assertEqual(
                    self.states(self.actual),
                    self.states(self.expected),
                )

    def test_callable_metadata_imports_copying_and_pickling_match(self):
        actual = self.actual
        expected = self.expected
        actual_function = actual.set_flags
        expected_function = expected.set_flags

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
        self.assertEqual(
            {
                name
                for name in vars(actual.m)
                if not name.startswith("_")
            },
            {
                name
                for name in vars(expected.m)
                if name
                in {
                    "ContextProp",
                    "CudnnModule",
                    "PropModule",
                    "contextmanager",
                    "flags",
                    "is_available",
                    "set_flags",
                    "torch",
                    "version",
                }
            },
        )

        self.assertIs(type(actual_function), types.FunctionType)
        self.assertIs(type(expected_function), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual_function)),
            str(inspect.signature(expected_function)),
        )
        self.assertEqual(
            inspect.get_annotations(actual_function),
            inspect.get_annotations(expected_function),
        )
        self.assertEqual(actual_function.__name__, expected_function.__name__)
        self.assertEqual(
            actual_function.__qualname__,
            expected_function.__qualname__,
        )
        self.assertEqual(
            actual_function.__module__.replace("torch_rs", "torch"),
            expected_function.__module__,
        )
        self.assertEqual(actual_function.__doc__, expected_function.__doc__)
        self.assertEqual(
            actual_function.__defaults__,
            expected_function.__defaults__,
        )
        self.assertEqual(
            actual_function.__kwdefaults__,
            expected_function.__kwdefaults__,
        )
        self.assertEqual(actual_function.__dict__, expected_function.__dict__)
        self.assertEqual(
            hasattr(actual_function, "__text_signature__"),
            hasattr(expected_function, "__text_signature__"),
        )
        self.assertEqual(
            actual_function.__code__.co_freevars,
            expected_function.__code__.co_freevars,
        )
        self.assertEqual(
            actual_function.__code__.co_cellvars,
            expected_function.__code__.co_cellvars,
        )

        for package_name, module in (
            ("torch_rs", actual),
            ("torch", expected),
        ):
            backend_import = {}
            function_import = {}
            wildcard = {}
            exec(f"from {package_name}.backends import cudnn", backend_import)
            exec(
                f"from {package_name}.backends.cudnn import set_flags",
                function_import,
            )
            exec(f"from {package_name}.backends.cudnn import *", wildcard)
            self.assertIs(backend_import["cudnn"], module)
            self.assertIs(function_import["set_flags"], module.set_flags)
            self.assertEqual(
                {name for name in wildcard if not name.startswith("__")},
                {"m"},
            )

        for function in (actual_function, expected_function):
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(actual_function, protocol)),
                    actual_function,
                )
                self.assertIs(
                    pickle.loads(pickle.dumps(expected_function, protocol)),
                    expected_function,
                )
                self.assertEqual(
                    self.pickle_shape(actual_function, protocol),
                    self.pickle_shape(expected_function, protocol),
                )

    def test_reload_stability_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_contract(torch, self.actual),
            self.reload_contract(reference_torch, self.expected),
        )

    def test_nonrepresentable_modes_fail_after_the_five_preferences(self):
        self.set_default(self.actual)
        with self.assertRaisesRegex(
            NotImplementedError,
            "^torch.backends.cudnn.flags\\(\\) only supports "
            "fp32_precision='none'$",
        ):
            self.actual.set_flags(*TARGET_STATE, "ieee", "cudnn")
        self.assertEqual(self.states(self.actual), TARGET_STATE)

        self.set_default(self.actual)
        with self.assertRaisesRegex(
            NotImplementedError,
            "^torch.backends.cudnn.flags\\(\\) only supports "
            "depthwise_kernel='auto'$",
        ):
            self.actual.set_flags(*TARGET_STATE, "none", "cudnn")
        self.assertEqual(self.states(self.actual), TARGET_STATE)

        self.set_default(self.expected)
        result = self.expected.set_flags(*TARGET_STATE, "ieee", "cudnn")
        self.assertEqual(result, (*DEFAULT_STATE, *DEFAULT_MODES))
        self.assertEqual(self.expected.fp32_precision, "ieee")
        self.assertEqual(self.expected.depthwise_kernel, "cudnn")


if __name__ == "__main__":
    unittest.main()
