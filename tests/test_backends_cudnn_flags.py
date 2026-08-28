import copy
import importlib
import inspect
import pickle
import re
import sys
import types
import unittest

import torch_rs as torch


DEFAULT_STATE = (True, False, 10, False, True)
TARGET_STATE = (False, True, 17, True, False)
FP32_ERROR = (
    "torch.backends.cudnn.flags() only supports fp32_precision='none'"
)
DEPTHWISE_ERROR = (
    "torch.backends.cudnn.flags() only supports depthwise_kernel='auto'"
)


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


class _BytearrayMode(bytearray):
    def __eq__(self, other):
        raise AssertionError("mode validation must not dispatch equality")

    def decode(self, *args, **kwargs):
        raise AssertionError("mode validation must not dispatch byte decoding")


def fresh_cudnn_module():
    module_name = "torch_rs.backends.cudnn"
    sys.modules.pop(module_name, None)
    if hasattr(torch.backends, "cudnn"):
        del torch.backends.cudnn
    module = importlib.import_module(module_name)
    torch.backends.cudnn = module
    return module


class CudnnFlagsTests(unittest.TestCase):
    def setUp(self):
        self.cudnn = fresh_cudnn_module()
        self.original = self.states(self.cudnn)
        self.set_states(self.cudnn, DEFAULT_STATE)

    def tearDown(self):
        self.set_states(fresh_cudnn_module(), self.original)

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

    def test_default_explicit_none_nested_and_exceptional_restoration(self):
        cudnn = self.cudnn
        initial_states = (
            DEFAULT_STATE,
            (False, True, -9, True, False),
        )

        for initial in initial_states:
            with self.subTest(initial=initial, mode="default"):
                self.set_states(cudnn, initial)
                context = cudnn.flags()
                self.assertEqual(self.states(cudnn), initial)
                with context as entered:
                    self.assertIsNone(entered)
                    self.assertEqual(
                        self.states(cudnn),
                        (False, False, 10, False, True),
                    )
                    self.assertIs(cudnn.is_available(), False)
                self.assertEqual(self.states(cudnn), initial)

            with self.subTest(initial=initial, mode="explicit"):
                self.set_states(cudnn, initial)
                context = cudnn.flags(*TARGET_STATE)
                self.assertEqual(self.states(cudnn), initial)
                self.assertIsNone(context.__enter__())
                self.assertEqual(self.states(cudnn), TARGET_STATE)
                self.assertIs(context.__exit__(None, None, None), False)
                self.assertEqual(self.states(cudnn), initial)

            with self.subTest(initial=initial, mode="none"):
                self.set_states(cudnn, initial)
                with cudnn.flags(None, None, None, None, None) as entered:
                    self.assertIsNone(entered)
                    self.assertEqual(self.states(cudnn), initial)
                self.assertEqual(self.states(cudnn), initial)

        self.set_states(cudnn, DEFAULT_STATE)
        outer_state = (False, True, 21, True, False)
        inner_state = (True, False, -7, False, True)
        with cudnn.flags(*outer_state) as outer:
            self.assertIsNone(outer)
            self.assertEqual(self.states(cudnn), outer_state)
            with cudnn.flags(*inner_state) as inner:
                self.assertIsNone(inner)
                self.assertEqual(self.states(cudnn), inner_state)
            self.assertEqual(self.states(cudnn), outer_state)
        self.assertEqual(self.states(cudnn), DEFAULT_STATE)

        marker = _ContextBodyError("body failed")
        with self.assertRaises(_ContextBodyError) as raised:
            with cudnn.flags(*TARGET_STATE) as entered:
                self.assertIsNone(entered)
                self.assertEqual(self.states(cudnn), TARGET_STATE)
                raise marker
        self.assertIs(raised.exception, marker)
        self.assertEqual(self.states(cudnn), DEFAULT_STATE)

        with cudnn.flags(*TARGET_STATE):
            self.set_states(cudnn, (True, False, 3, False, True))
        self.assertEqual(self.states(cudnn), DEFAULT_STATE)

    def test_failed_entry_mutates_left_to_right_without_rollback(self):
        cases = (
            (
                (1, True, 17, True, False),
                RuntimeError,
                "set_enabled_cudnn expects a bool, but got int",
                DEFAULT_STATE,
            ),
            (
                (False, 1, 17, True, False),
                RuntimeError,
                "set_benchmark_cudnn expects a bool, but got int",
                (False, False, 10, False, True),
            ),
            (
                (False, True, "invalid", True, False),
                RuntimeError,
                "set_benchmark_limit_cudnn expects an int, but got str",
                (False, True, 10, False, True),
            ),
            (
                (False, True, 17, 1, False),
                RuntimeError,
                "set_deterministic_cudnn expects a bool, but got int",
                (False, True, 17, False, True),
            ),
            (
                (False, True, 17, True, 1),
                RuntimeError,
                "set_allow_tf32_cublas expects a bool, but got int",
                (False, True, 17, True, True),
            ),
        )

        for arguments, error_type, message, expected_state in cases:
            with self.subTest(arguments=arguments):
                self.set_states(self.cudnn, DEFAULT_STATE)
                context = self.cudnn.flags(*arguments)
                self.assertEqual(self.states(self.cudnn), DEFAULT_STATE)
                with self.assertRaises(error_type) as raised:
                    context.__enter__()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertEqual(self.states(self.cudnn), expected_state)

        self.set_states(self.cudnn, DEFAULT_STATE)
        context = self.cudnn.flags(
            False,
            1,
            17,
            True,
            False,
            fp32_precision="ieee",
            depthwise_kernel="cudnn",
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "^set_benchmark_cudnn expects a bool, but got int$",
        ):
            context.__enter__()
        self.assertEqual(
            self.states(self.cudnn),
            (False, False, 10, False, True),
        )

    def test_trailing_modes_accept_defaults_and_reject_nondefaults_last(self):
        default_modes = (
            ("none", "auto"),
            (b"none", b"auto"),
            (_StringMode("none"), _StringMode("auto")),
            (_BytesMode(b"none"), _BytesMode(b"auto")),
            (bytearray(b"none"), "auto"),
            (_BytearrayMode(b"none"), "auto"),
            (None, None),
        )
        for fp32_precision, depthwise_kernel in default_modes:
            with self.subTest(
                fp32_precision=type(fp32_precision).__name__,
                depthwise_kernel=type(depthwise_kernel).__name__,
            ):
                with self.cudnn.flags(
                    *TARGET_STATE,
                    fp32_precision=fp32_precision,
                    depthwise_kernel=depthwise_kernel,
                ) as entered:
                    self.assertIsNone(entered)
                    self.assertEqual(self.states(self.cudnn), TARGET_STATE)
                self.assertEqual(self.states(self.cudnn), DEFAULT_STATE)

        cases = (
            (
                {"fp32_precision": "ieee"},
                FP32_ERROR,
            ),
            (
                {"fp32_precision": "tf32"},
                FP32_ERROR,
            ),
            (
                {"depthwise_kernel": "cudnn"},
                DEPTHWISE_ERROR,
            ),
            (
                {"depthwise_kernel": "native"},
                DEPTHWISE_ERROR,
            ),
            (
                {"depthwise_kernel": bytearray(b"auto")},
                "set_cudnn_depthwise_kernel expects a string, but got "
                "bytearray",
            ),
            (
                {"depthwise_kernel": _BytearrayMode(b"auto")},
                "set_cudnn_depthwise_kernel expects a string, but got "
                "_BytearrayMode",
            ),
            (
                {
                    "fp32_precision": "ieee",
                    "depthwise_kernel": "cudnn",
                },
                FP32_ERROR,
            ),
        )
        for keywords, message in cases:
            with self.subTest(keywords=keywords):
                self.set_states(self.cudnn, DEFAULT_STATE)
                context = self.cudnn.flags(*TARGET_STATE, **keywords)
                self.assertEqual(self.states(self.cudnn), DEFAULT_STATE)
                error_type = (
                    RuntimeError
                    if isinstance(keywords.get("depthwise_kernel"), bytearray)
                    else NotImplementedError
                )
                with self.assertRaises(error_type) as raised:
                    context.__enter__()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertEqual(self.states(self.cudnn), TARGET_STATE)

        self.assertFalse(hasattr(self.cudnn, "fp32_precision"))
        self.assertFalse(hasattr(self.cudnn, "depthwise_kernel"))

    def test_context_is_a_reusable_decorator_factory(self):
        observations = []

        @self.cudnn.flags(*TARGET_STATE)
        def decorated(value):
            observations.append(self.states(self.cudnn))
            if value == "raise":
                raise _ContextBodyError("decorated body failed")
            return value

        self.assertEqual(decorated("first"), "first")
        self.assertEqual(self.states(self.cudnn), DEFAULT_STATE)
        self.assertEqual(decorated("second"), "second")
        self.assertEqual(self.states(self.cudnn), DEFAULT_STATE)
        with self.assertRaisesRegex(_ContextBodyError, "decorated body failed"):
            decorated("raise")
        self.assertEqual(
            observations,
            [TARGET_STATE, TARGET_STATE, TARGET_STATE],
        )
        self.assertEqual(self.states(self.cudnn), DEFAULT_STATE)

    def test_metadata_imports_copying_and_pickling(self):
        cudnn = self.cudnn
        function = cudnn.flags
        wrapped = function.__wrapped__

        self.assertIs(torch.backends.cudnn, cudnn)
        self.assertIs(sys.modules["torch_rs.backends.cudnn"], cudnn)
        self.assertEqual(type(cudnn).__name__, "CudnnModule")
        self.assertIsNone(cudnn.__doc__)
        self.assertFalse(hasattr(cudnn, "__all__"))
        self.assertEqual(
            {name for name in vars(cudnn) if not name.startswith("_")},
            {"m"},
        )
        self.assertEqual(
            {name for name in vars(cudnn.m) if not name.startswith("_")},
            {
                "CudnnModule",
                "contextmanager",
                "flags",
                "is_available",
                "set_flags",
                "torch",
                "version",
            },
        )
        self.assertIs(function, cudnn.m.flags)
        self.assertIs(cudnn.set_flags, cudnn.m.set_flags)

        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(function)),
            "(enabled=False, benchmark=False, benchmark_limit=10, "
            "deterministic=False, allow_tf32=True, fp32_precision='none', "
            "depthwise_kernel='auto')",
        )
        self.assertEqual(inspect.get_annotations(function), {})
        self.assertEqual(function.__name__, "flags")
        self.assertEqual(function.__qualname__, "flags")
        self.assertEqual(function.__module__, "torch_rs.backends.cudnn")
        self.assertIs(inspect.getmodule(function), cudnn)
        self.assertIsNone(function.__doc__)
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {"__wrapped__": wrapped})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(function.__code__.co_freevars, ("func",))
        self.assertEqual(function.__code__.co_cellvars, ())

        self.assertIs(type(wrapped), types.FunctionType)
        self.assertEqual(inspect.signature(wrapped), inspect.signature(function))
        self.assertEqual(inspect.get_annotations(wrapped), {})
        self.assertEqual(wrapped.__name__, "flags")
        self.assertEqual(wrapped.__qualname__, "flags")
        self.assertEqual(wrapped.__module__, "torch_rs.backends.cudnn")
        self.assertIsNone(wrapped.__doc__)
        self.assertEqual(
            wrapped.__defaults__,
            (False, False, 10, False, True, "none", "auto"),
        )
        self.assertIsNone(wrapped.__kwdefaults__)
        self.assertEqual(wrapped.__dict__, {})
        self.assertEqual(
            wrapped.__code__.co_names,
            ("__allow_nonbracketed_mutation", "set_flags"),
        )
        self.assertEqual(wrapped.__code__.co_freevars, ())
        self.assertEqual(wrapped.__code__.co_cellvars, ())

        backend_import = {}
        function_import = {}
        child_wildcard = {}
        exec("from torch_rs.backends import cudnn", backend_import)
        exec("from torch_rs.backends.cudnn import flags", function_import)
        exec("from torch_rs.backends.cudnn import *", child_wildcard)
        self.assertIs(backend_import["cudnn"], cudnn)
        self.assertIs(function_import["flags"], function)
        self.assertEqual(
            {name for name in child_wildcard if not name.startswith("__")},
            {"m"},
        )
        self.assertIs(child_wildcard["m"], cudnn.m)
        setter_import = {}
        exec("from torch_rs.backends.cudnn import set_flags", setter_import)
        self.assertIs(setter_import["set_flags"], cudnn.set_flags)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(kind="function", protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.backends.cudnn", payload)
                self.assertIs(pickle.loads(payload), function)

        context = function(*TARGET_STATE)
        self.assertEqual(type(context).__module__, "contextlib")
        self.assertEqual(type(context).__qualname__, "_GeneratorContextManager")
        self.assertEqual(context.__doc__, "Helper for @contextmanager decorator.")
        self.assertIs(context.func, wrapped)
        self.assertEqual(context.args, TARGET_STATE)
        self.assertEqual(context.kwds, {})
        copied = copy.copy(context)
        self.assertIsNot(copied, context)
        self.assertIs(copied.gen, context.gen)
        with self.assertRaisesRegex(TypeError, "cannot pickle 'generator' object"):
            copy.deepcopy(context)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(kind="context", protocol=protocol):
                with self.assertRaisesRegex(
                    TypeError,
                    "cannot pickle 'generator' object",
                ):
                    pickle.dumps(context, protocol=protocol)

    def test_reload_preserves_state_and_existing_contexts(self):
        cudnn = self.cudnn
        old_flags = cudnn.flags
        old_wrapped = old_flags.__wrapped__
        namespace = cudnn.__dict__
        active_context = old_flags(*TARGET_STATE)

        self.assertIsNone(active_context.__enter__())
        self.assertEqual(self.states(cudnn), TARGET_STATE)
        reloaded = importlib.reload(cudnn)

        self.assertIsNot(reloaded, cudnn)
        self.assertIs(cudnn.__dict__, namespace)
        self.assertIs(torch.backends.cudnn, cudnn)
        self.assertIs(sys.modules[cudnn.__name__], reloaded)
        self.assertIs(reloaded.m, cudnn)
        self.assertIsNot(cudnn.flags, old_flags)
        self.assertIsNot(cudnn.flags.__wrapped__, old_wrapped)
        self.assertEqual(self.states(cudnn), TARGET_STATE)
        self.assertIs(active_context.__exit__(None, None, None), False)
        self.assertEqual(self.states(cudnn), DEFAULT_STATE)

        for function in (old_flags, cudnn.flags):
            with self.subTest(function=function):
                with function(*TARGET_STATE) as entered:
                    self.assertIsNone(entered)
                    self.assertEqual(self.states(cudnn), TARGET_STATE)
                self.assertEqual(self.states(cudnn), DEFAULT_STATE)

        with self.assertRaises(pickle.PicklingError) as raised:
            pickle.dumps(old_flags)
        message = re.sub(r"0x[0-9a-fA-F]+", "0x...", str(raised.exception))
        self.assertEqual(
            message,
            "Can't pickle <function flags at 0x...>: it's not the same object "
            "as torch_rs.backends.cudnn.flags",
        )
        self.assertIs(
            pickle.loads(pickle.dumps(cudnn.flags)),
            cudnn.flags,
        )
        self.assertIs(cudnn.set_flags, reloaded.set_flags)

    def test_context_management_does_not_add_cudnn_execution(self):
        self.assertIs(self.cudnn.is_available(), False)
        with self.cudnn.flags(*TARGET_STATE):
            self.assertIs(self.cudnn.is_available(), False)
            self.assertIs(self.cudnn.version(), None)
            self.assertFalse(hasattr(torch, "cudnn_convolution"))
            self.assertFalse(hasattr(torch, "cuda"))
        self.assertIs(self.cudnn.is_available(), False)


if __name__ == "__main__":
    unittest.main()
