import copy
import importlib
import inspect
import json
import os
import pickle
import pickletools
import subprocess
import sys
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


BACKEND_MODULES = {
    "cpu",
    "cuda",
    "cudnn",
    "kleidiai",
    "mha",
    "mkl",
    "nnpack",
    "openmp",
}


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class BackendsGlobalFlagsReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "backends global-flag differentials require pinned " "PyTorch 2.13.0"
            )

    def normalize(self, value):
        if isinstance(value, str):
            return value.replace("torch_rs", "torch")
        return value

    def pickle_shape(self, value, protocol):
        shape = []
        for opcode, argument, _ in pickletools.genops(
            pickle.dumps(value, protocol=protocol)
        ):
            if opcode.name == "FRAME":
                argument = "<frame length>"
            elif isinstance(argument, str):
                argument = self.normalize(argument)
            shape.append((opcode.name, argument))
        return shape

    def test_metadata_imports_wildcards_copying_and_pickling_match(self):
        actual = importlib.import_module("torch_rs.backends")
        expected = importlib.import_module("torch.backends")

        self.assertEqual(type(actual).__name__, type(expected).__name__)
        self.assertEqual(
            self.normalize(type(actual).__module__),
            type(expected).__module__,
        )
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(hasattr(actual, "__all__"), hasattr(expected, "__all__"))
        self.assertIs(type(actual.m), types.ModuleType)
        self.assertIs(type(expected.m), types.ModuleType)

        supported_proxy_names = {*BACKEND_MODULES, "m"}
        self.assertEqual(
            {name for name in vars(actual) if not name.startswith("_")},
            {name for name in vars(expected) if name in supported_proxy_names},
        )

        actual_wildcard = {}
        expected_wildcard = {}
        exec("from torch_rs.backends import *", actual_wildcard)
        exec("from torch.backends import *", expected_wildcard)
        self.assertEqual(
            {name for name in actual_wildcard if not name.startswith("__")},
            {name for name in expected_wildcard if name in supported_proxy_names},
        )

        for name in ("disable_global_flags", "flags_frozen"):
            actual_function = getattr(actual, name)
            expected_function = getattr(expected, name)
            with self.subTest(function=name):
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
                    self.normalize(actual_function.__module__),
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
                    actual_function.__code__.co_names,
                    expected_function.__code__.co_names,
                )
                self.assertEqual(
                    actual_function.__code__.co_freevars,
                    expected_function.__code__.co_freevars,
                )
                self.assertEqual(
                    actual_function.__code__.co_cellvars,
                    expected_function.__code__.co_cellvars,
                )

                for package_name, function in (
                    ("torch_rs", actual_function),
                    ("torch", expected_function),
                ):
                    namespace = {}
                    exec(
                        f"from {package_name}.backends import {name}",
                        namespace,
                    )
                    self.assertIs(namespace[name], function)

                self.assertNotIn(name, actual_wildcard)
                self.assertNotIn(name, expected_wildcard)
                for function in (actual_function, expected_function):
                    self.assertIs(copy.copy(function), function)
                    self.assertIs(copy.deepcopy(function), function)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
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

        actual_descriptor = vars(type(actual.cudnn))["enabled"]
        expected_descriptor = vars(type(expected.cudnn))["enabled"]
        self.assertEqual(
            self.normalize(type(actual_descriptor).__module__),
            type(expected_descriptor).__module__,
        )
        self.assertEqual(
            type(actual_descriptor).__name__,
            type(expected_descriptor).__name__,
        )
        self.assertEqual(
            set(vars(actual_descriptor)),
            set(vars(expected_descriptor)),
        )
        for descriptor in (actual_descriptor, expected_descriptor):
            for copier in (copy.copy, copy.deepcopy):
                copied = copier(descriptor)
                self.assertIsNot(copied, descriptor)
                self.assertIs(type(copied), type(descriptor))
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                restored = pickle.loads(pickle.dumps(descriptor, protocol))
                self.assertIs(type(restored), type(descriptor))

    def test_argument_errors_match(self):
        for name in ("disable_global_flags", "flags_frozen"):
            actual = getattr(torch.backends, name)
            expected = getattr(reference_torch.backends, name)
            cases = (
                (lambda: actual(None), lambda: expected(None)),
                (lambda: actual(None, None), lambda: expected(None, None)),
                (
                    lambda: actual(enabled=True),
                    lambda: expected(enabled=True),
                ),
            )
            for case, (actual_call, expected_call) in enumerate(cases):
                with self.subTest(function=name, case=case):
                    with self.assertRaises(Exception) as actual_raised:
                        actual_call()
                    with self.assertRaises(Exception) as expected_raised:
                        expected_call()
                    self.assertIs(
                        type(actual_raised.exception),
                        type(expected_raised.exception),
                    )
                    self.assertEqual(
                        self.normalize(str(actual_raised.exception)),
                        str(expected_raised.exception),
                    )
                    self.assertEqual(
                        tuple(
                            self.normalize(argument)
                            for argument in actual_raised.exception.args
                        ),
                        expected_raised.exception.args,
                    )

    def behavioral_contract(self, package_name):
        script = r"""
import importlib
import json
import pickle
import re
import sys
import threading

root = importlib.import_module(sys.argv[1])
backends = root.backends
cudnn = backends.cudnn
names = ("enabled", "benchmark", "benchmark_limit", "deterministic", "allow_tf32")
initial = (True, False, 10, False, True)
target = (False, True, 17, True, False)
cudnn.set_flags(*initial)
result = {"initial": backends.flags_frozen()}

thread_values = []
def freeze():
    thread_values.extend(
        (
            backends.flags_frozen(),
            backends.disable_global_flags(),
            backends.flags_frozen(),
        )
    )

thread = threading.Thread(target=freeze)
thread.start()
thread.join()
result["thread"] = thread_values
result["frozen"] = backends.flags_frozen()
result["repeat"] = backends.disable_global_flags()

assignment_errors = []
for name, value in zip(names, target):
    before = getattr(cudnn, name)
    try:
        setattr(cudnn, name, value)
    except Exception as error:
        assignment_errors.append(
            (type(error).__name__, str(error), error.args, getattr(cudnn, name), before)
        )
    else:
        raise AssertionError(f"{name} assignment unexpectedly succeeded")
result["assignments"] = assignment_errors

result["set_flags"] = (
    cudnn.set_flags(*target),
    tuple(getattr(cudnn, name) for name in names),
    backends.flags_frozen(),
)
with cudnn.flags(*initial) as entered:
    inside = (
        entered,
        tuple(getattr(cudnn, name) for name in names),
        backends.flags_frozen(),
    )
    try:
        cudnn.enabled = False
    except Exception as error:
        inside_error = (type(error).__name__, str(error), error.args)
    else:
        raise AssertionError("assignment inside flags() unexpectedly succeeded")
result["context"] = (
    inside,
    inside_error,
    tuple(getattr(cudnn, name) for name in names),
    backends.flags_frozen(),
)

reloaded_cudnn = importlib.reload(cudnn)
cudnn_reload_errors = []
for module in (cudnn, reloaded_cudnn):
    try:
        module.enabled = False
    except Exception as error:
        cudnn_reload_errors.append((type(error).__name__, str(error), error.args))
    else:
        raise AssertionError("a reloaded cuDNN proxy unexpectedly became mutable")
result["cudnn_reload"] = (
    reloaded_cudnn is not cudnn,
    backends.cudnn is cudnn,
    sys.modules[f"{root.__name__}.backends.cudnn"] is reloaded_cudnn,
    reloaded_cudnn.m is cudnn,
    backends.flags_frozen(),
    cudnn_reload_errors,
)

old_disable = backends.disable_global_flags
old_frozen = backends.flags_frozen
reloaded = importlib.reload(backends)
try:
    cudnn.enabled = False
except Exception as error:
    stale_cudnn_error = (type(error).__name__, str(error), error.args)
else:
    raise AssertionError("the stale cuDNN proxy unexpectedly became mutable")

stale_pickle_errors = []
for function in (old_disable, old_frozen):
    try:
        pickle.dumps(function)
    except Exception as error:
        stale_pickle_errors.append(
            (
                type(error).__name__,
                re.sub(r"0x[0-9a-fA-F]+", "0x...", str(error)),
            )
        )
    else:
        raise AssertionError("a stale backend function remained pickleable")

result["reload"] = (
    reloaded is not backends,
    root.backends is backends,
    sys.modules[f"{root.__name__}.backends"] is reloaded,
    reloaded.m is backends,
    reloaded.flags_frozen(),
    backends.flags_frozen(),
    old_frozen(),
    stale_cudnn_error,
    stale_pickle_errors,
    pickle.loads(pickle.dumps(reloaded.disable_global_flags))
        is reloaded.disable_global_flags,
    pickle.loads(pickle.dumps(reloaded.flags_frozen)) is reloaded.flags_frozen,
)
print(json.dumps(result, sort_keys=True))
"""
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ""
        completed = subprocess.run(
            [sys.executable, "-c", script, package_name],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )
        return json.loads(self.normalize(completed.stdout))

    def test_state_assignment_context_thread_and_reload_behavior_match(self):
        self.assertEqual(
            self.behavioral_contract("torch_rs"),
            self.behavioral_contract("torch"),
        )


if __name__ == "__main__":
    unittest.main()
