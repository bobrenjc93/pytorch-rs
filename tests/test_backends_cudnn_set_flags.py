import copy
import importlib
import inspect
import json
import pickle
import re
import subprocess
import sys
import threading
import types
import unittest

import torch_rs as torch


DEFAULT_STATE = (True, False, 10, False, True)
TARGET_STATE = (False, True, 17, True, False)
DEFAULT_MODES = ("none", "auto")
FP32_ERROR = (
    "torch.backends.cudnn.flags() only supports fp32_precision='none'"
)
DEPTHWISE_ERROR = (
    "torch.backends.cudnn.flags() only supports depthwise_kernel='auto'"
)


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


class CudnnSetFlagsTests(unittest.TestCase):
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

    def test_fresh_process_returns_seven_previous_values_without_cudnn(self):
        script = r'''
import json

import torch_rs as torch

cudnn = torch.backends.cudnn
first = cudnn.set_flags()
second = cudnn.set_flags(False, True, 17, True, False)
third = cudnn.set_flags(None, None, None, None, None, None, None)
print(json.dumps({
    "available": cudnn.is_available(),
    "first": first,
    "second": second,
    "third": third,
    "state": [
        cudnn.enabled,
        cudnn.benchmark,
        cudnn.benchmark_limit,
        cudnn.deterministic,
        cudnn.allow_tf32,
    ],
    "execution": hasattr(torch, "cudnn_convolution"),
}))
'''
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )
        self.assertEqual(
            json.loads(completed.stdout),
            {
                "available": False,
                "first": [*DEFAULT_STATE, *DEFAULT_MODES],
                "second": [*DEFAULT_STATE, *DEFAULT_MODES],
                "third": [*TARGET_STATE, *DEFAULT_MODES],
                "state": list(TARGET_STATE),
                "execution": False,
            },
        )

    def test_updates_return_exact_previous_state_and_none_skips_fields(self):
        result = self.cudnn.set_flags(*TARGET_STATE)
        self.assertIs(type(result), tuple)
        self.assertEqual(len(result), 7)
        self.assertEqual(result, (*DEFAULT_STATE, *DEFAULT_MODES))
        self.assertEqual(self.states(self.cudnn), TARGET_STATE)
        for value, expected_type in zip(
            result,
            (bool, bool, int, bool, bool, str, str),
        ):
            self.assertIs(type(value), expected_type)

        result = self.cudnn.set_flags(
            None,
            False,
            None,
            None,
            True,
            None,
            None,
        )
        self.assertEqual(result, (*TARGET_STATE, *DEFAULT_MODES))
        self.assertEqual(
            self.states(self.cudnn),
            (False, False, 17, True, True),
        )

        result = self.cudnn.set_flags(
            _enabled=True,
            _benchmark=None,
            _benchmark_limit=10,
            _deterministic=False,
            _allow_tf32=None,
            _fp32_precision="none",
            _depthwise_kernel="auto",
        )
        self.assertEqual(
            result,
            (False, False, 17, True, True, *DEFAULT_MODES),
        )
        self.assertEqual(self.states(self.cudnn), DEFAULT_STATE)

    def test_failed_updates_mutate_left_to_right_without_rollback(self):
        cases = (
            (
                (1, True, 17, True, False),
                "set_enabled_cudnn expects a bool, but got int",
                DEFAULT_STATE,
            ),
            (
                (False, 1, 17, True, False),
                "set_benchmark_cudnn expects a bool, but got int",
                (False, False, 10, False, True),
            ),
            (
                (False, True, "invalid", True, False),
                "set_benchmark_limit_cudnn expects an int, but got str",
                (False, True, 10, False, True),
            ),
            (
                (False, True, 17, 1, False),
                "set_deterministic_cudnn expects a bool, but got int",
                (False, True, 17, False, True),
            ),
            (
                (False, True, 17, True, 1),
                "set_allow_tf32_cublas expects a bool, but got int",
                (False, True, 17, True, True),
            ),
        )
        for arguments, message, expected_state in cases:
            with self.subTest(arguments=arguments):
                self.set_states(self.cudnn, DEFAULT_STATE)
                with self.assertRaises(RuntimeError) as raised:
                    self.cudnn.set_flags(*arguments)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertEqual(self.states(self.cudnn), expected_state)

        self.set_states(self.cudnn, DEFAULT_STATE)
        with self.assertRaisesRegex(
            RuntimeError,
            "^set_benchmark_cudnn expects a bool, but got int$",
        ):
            self.cudnn.set_flags(
                False,
                1,
                17,
                True,
                False,
                "ieee",
                "cudnn",
            )
        self.assertEqual(
            self.states(self.cudnn),
            (False, False, 10, False, True),
        )

    def test_only_representable_trailing_modes_are_accepted(self):
        supported_modes = (
            ("none", "auto"),
            (b"none", b"auto"),
            (_StringMode("none"), _StringMode("auto")),
            (_BytesMode(b"none"), _BytesMode(b"auto")),
            (bytearray(b"none"), None),
            (_BytearrayMode(b"none"), None),
            (None, None),
        )
        for fp32_precision, depthwise_kernel in supported_modes:
            with self.subTest(
                fp32_precision=type(fp32_precision).__name__,
                depthwise_kernel=type(depthwise_kernel).__name__,
            ):
                self.set_states(self.cudnn, DEFAULT_STATE)
                result = self.cudnn.set_flags(
                    *TARGET_STATE,
                    fp32_precision,
                    depthwise_kernel,
                )
                self.assertEqual(result, (*DEFAULT_STATE, *DEFAULT_MODES))
                self.assertEqual(self.states(self.cudnn), TARGET_STATE)

        cases = (
            ("ieee", "auto", NotImplementedError, FP32_ERROR),
            ("tf32", None, NotImplementedError, FP32_ERROR),
            ("none", "cudnn", NotImplementedError, DEPTHWISE_ERROR),
            (None, "native", NotImplementedError, DEPTHWISE_ERROR),
            (
                "none",
                bytearray(b"auto"),
                RuntimeError,
                "set_cudnn_depthwise_kernel expects a string, but got bytearray",
            ),
            (
                "none",
                _BytearrayMode(b"auto"),
                RuntimeError,
                "set_cudnn_depthwise_kernel expects a string, but got "
                "_BytearrayMode",
            ),
            ("ieee", "cudnn", NotImplementedError, FP32_ERROR),
        )
        for fp32_precision, depthwise_kernel, error_type, message in cases:
            with self.subTest(
                fp32_precision=fp32_precision,
                depthwise_kernel=depthwise_kernel,
            ):
                self.set_states(self.cudnn, DEFAULT_STATE)
                with self.assertRaises(error_type) as raised:
                    self.cudnn.set_flags(
                        *TARGET_STATE,
                        fp32_precision,
                        depthwise_kernel,
                    )
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertEqual(self.states(self.cudnn), TARGET_STATE)

        self.assertFalse(hasattr(self.cudnn, "fp32_precision"))
        self.assertFalse(hasattr(self.cudnn, "depthwise_kernel"))

    def test_state_is_process_global_across_threads(self):
        cudnn = self.cudnn
        worker_changed = threading.Event()
        main_changed = threading.Event()
        observations = []
        errors = []

        def worker():
            try:
                observations.append(cudnn.set_flags(*TARGET_STATE))
                worker_changed.set()
                if not main_changed.wait(timeout=10):
                    raise RuntimeError("timed out waiting for main-thread update")
                observations.append(cudnn.set_flags(*TARGET_STATE))
            except BaseException as error:
                errors.append(error)
                worker_changed.set()

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(worker_changed.wait(timeout=10))
        self.assertEqual(errors, [])
        self.assertEqual(self.states(cudnn), TARGET_STATE)
        self.assertEqual(
            cudnn.set_flags(*DEFAULT_STATE),
            (*TARGET_STATE, *DEFAULT_MODES),
        )
        main_changed.set()
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(
            observations,
            [
                (*DEFAULT_STATE, *DEFAULT_MODES),
                (*DEFAULT_STATE, *DEFAULT_MODES),
            ],
        )
        self.assertEqual(self.states(cudnn), TARGET_STATE)

    def test_metadata_imports_copying_and_pickling(self):
        cudnn = self.cudnn
        function = cudnn.set_flags

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
                "ContextProp",
                "CudnnModule",
                "PropModule",
                "contextmanager",
                "flags",
                "is_available",
                "set_flags",
                "torch",
                "version",
            },
        )
        self.assertIs(function, cudnn.m.set_flags)

        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(function)),
            "(_enabled=None, _benchmark=None, _benchmark_limit=None, "
            "_deterministic=None, _allow_tf32=None, "
            "_fp32_precision='none', _depthwise_kernel=None)",
        )
        self.assertEqual(inspect.get_annotations(function), {})
        self.assertEqual(function.__name__, "set_flags")
        self.assertEqual(function.__qualname__, "set_flags")
        self.assertEqual(function.__module__, "torch_rs.backends.cudnn")
        self.assertIs(inspect.getmodule(function), cudnn)
        self.assertIsNone(function.__doc__)
        self.assertEqual(
            function.__defaults__,
            (None, None, None, None, None, "none", None),
        )
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

        backend_import = {}
        function_import = {}
        child_wildcard = {}
        exec("from torch_rs.backends import cudnn", backend_import)
        exec("from torch_rs.backends.cudnn import set_flags", function_import)
        exec("from torch_rs.backends.cudnn import *", child_wildcard)
        self.assertIs(backend_import["cudnn"], cudnn)
        self.assertIs(function_import["set_flags"], function)
        self.assertEqual(
            {name for name in child_wildcard if not name.startswith("__")},
            {"m"},
        )
        self.assertIs(child_wildcard["m"], cudnn.m)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.backends.cudnn", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_reload_preserves_state_and_replaces_the_public_setter(self):
        cudnn = self.cudnn
        old_setter = cudnn.set_flags
        namespace = cudnn.__dict__

        self.assertEqual(
            old_setter(*TARGET_STATE),
            (*DEFAULT_STATE, *DEFAULT_MODES),
        )
        reloaded = importlib.reload(cudnn)

        self.assertIsNot(reloaded, cudnn)
        self.assertIs(cudnn.__dict__, namespace)
        self.assertIs(torch.backends.cudnn, cudnn)
        self.assertIs(sys.modules[cudnn.__name__], reloaded)
        self.assertIs(reloaded.m, cudnn)
        self.assertIsNot(cudnn.set_flags, old_setter)
        self.assertIs(reloaded.set_flags, cudnn.set_flags)
        self.assertEqual(self.states(cudnn), TARGET_STATE)
        self.assertEqual(
            cudnn.set_flags(*DEFAULT_STATE),
            (*TARGET_STATE, *DEFAULT_MODES),
        )
        self.assertEqual(
            old_setter(*TARGET_STATE),
            (*DEFAULT_STATE, *DEFAULT_MODES),
        )
        self.assertEqual(
            cudnn.set_flags(*DEFAULT_STATE),
            (*TARGET_STATE, *DEFAULT_MODES),
        )

        with self.assertRaises(pickle.PicklingError) as raised:
            pickle.dumps(old_setter)
        message = re.sub(r"0x[0-9a-fA-F]+", "0x...", str(raised.exception))
        self.assertEqual(
            message,
            "Can't pickle <function set_flags at 0x...>: it's not the same "
            "object as torch_rs.backends.cudnn.set_flags",
        )
        self.assertIs(
            pickle.loads(pickle.dumps(cudnn.set_flags)),
            cudnn.set_flags,
        )

    def test_call_binding_errors_leave_state_unchanged(self):
        unexpected_keyword = (
            "set_flags() got an unexpected keyword argument 'enabled'"
        )
        if sys.version_info >= (3, 13):
            unexpected_keyword += ". Did you mean '_enabled'?"
        cases = (
            (
                lambda: self.cudnn.set_flags(
                    None,
                    None,
                    None,
                    None,
                    None,
                    "none",
                    None,
                    None,
                ),
                "set_flags() takes from 0 to 7 positional arguments but 8 "
                "were given",
            ),
            (
                lambda: self.cudnn.set_flags(enabled=True),
                unexpected_keyword,
            ),
            (
                lambda: self.cudnn.set_flags(False, _enabled=True),
                "set_flags() got multiple values for argument '_enabled'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertEqual(self.states(self.cudnn), DEFAULT_STATE)

    def test_public_setter_does_not_add_cudnn_execution(self):
        self.assertIs(self.cudnn.is_available(), False)
        self.assertEqual(
            self.cudnn.set_flags(*TARGET_STATE),
            (*DEFAULT_STATE, *DEFAULT_MODES),
        )
        self.assertIs(self.cudnn.is_available(), False)
        self.assertIs(self.cudnn.version(), None)
        self.assertFalse(hasattr(torch, "cudnn_convolution"))
        self.assertIs(torch.cuda.is_available(), False)
        self.assertEqual(torch.cuda.device_count(), 0)


if __name__ == "__main__":
    unittest.main()
