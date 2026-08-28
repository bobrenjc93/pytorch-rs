import copy
import importlib
import inspect
import pickle
import re
import subprocess
import sys
import types
import unittest

import torch_rs as torch


BACKEND_MODULES = {
    "cpu",
    "cuda",
    "cudnn",
    "cusparselt",
    "kleidiai",
    "mha",
    "mkl",
    "nnpack",
    "openmp",
}
FLAG_NAMES = (
    "enabled",
    "benchmark",
    "benchmark_limit",
    "deterministic",
    "allow_tf32",
)
DEFAULT_STATE = (True, False, 10, False, True)
TARGET_STATE = (False, True, 17, True, False)
FROZEN_ERROR = (
    "not allowed to set torch_rs.backends.cudnn flags after "
    "disable_global_flags; please use flags() context manager instead"
)


class BackendsGlobalFlagsTests(unittest.TestCase):
    def run_script(self, script):
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

    def test_initial_state_one_way_transition_and_thread_visibility(self):
        self.run_script(
            r"""
import threading

import torch_rs as torch

backends = torch.backends
assert backends.flags_frozen() is False
observed = []

def freeze():
    observed.append(backends.flags_frozen())
    observed.append(backends.disable_global_flags())
    observed.append(backends.flags_frozen())

thread = threading.Thread(target=freeze)
thread.start()
thread.join()
assert observed == [False, None, True]
assert backends.flags_frozen() is True

seen_from_another_thread = []
thread = threading.Thread(
    target=lambda: seen_from_another_thread.append(backends.flags_frozen())
)
thread.start()
thread.join()
assert seen_from_another_thread == [True]
assert backends.disable_global_flags() is None
assert backends.flags_frozen() is True
"""
        )

    def test_frozen_cudnn_assignments_fail_without_mutation(self):
        self.run_script(
            f"""
import torch_rs as torch

cudnn = torch.backends.cudnn
names = {FLAG_NAMES!r}
initial = {DEFAULT_STATE!r}
targets = {TARGET_STATE!r}
cudnn.set_flags(*initial)
torch.backends.disable_global_flags()

for name, target, before in zip(names, targets, initial):
    try:
        setattr(cudnn, name, target)
    except RuntimeError as error:
        assert str(error) == {FROZEN_ERROR!r}
        assert error.args == ({FROZEN_ERROR!r},)
    else:
        raise AssertionError(f"{{name}} assignment unexpectedly succeeded")
    assert getattr(cudnn, name) == before

assert tuple(getattr(cudnn, name) for name in names) == initial
assert torch.backends.flags_frozen() is True
"""
        )

    def test_set_flags_and_flags_keep_their_frozen_behavior(self):
        self.run_script(
            f"""
import torch_rs as torch

cudnn = torch.backends.cudnn
names = {FLAG_NAMES!r}
initial = {DEFAULT_STATE!r}
target = {TARGET_STATE!r}
cudnn.set_flags(*initial)
torch.backends.disable_global_flags()

assert cudnn.set_flags(*target) == (*initial, "none", "auto")
assert tuple(getattr(cudnn, name) for name in names) == target
assert torch.backends.flags_frozen() is True

with cudnn.flags(*initial) as entered:
    assert entered is None
    assert tuple(getattr(cudnn, name) for name in names) == initial
    assert torch.backends.flags_frozen() is True
    try:
        cudnn.enabled = False
    except RuntimeError as error:
        assert str(error) == {FROZEN_ERROR!r}
    else:
        raise AssertionError("assignment inside flags() unexpectedly succeeded")

assert tuple(getattr(cudnn, name) for name in names) == target
assert torch.backends.flags_frozen() is True

class MarkerError(Exception):
    pass

try:
    with cudnn.flags(*initial):
        raise MarkerError
except MarkerError:
    pass
else:
    raise AssertionError("flags() suppressed the body exception")

assert tuple(getattr(cudnn, name) for name in names) == target
assert torch.backends.flags_frozen() is True
"""
        )

    def test_callable_and_proxy_metadata_imports_copying_and_pickling(self):
        backends = importlib.import_module("torch_rs.backends")
        functions = {
            "disable_global_flags": backends.disable_global_flags,
            "flags_frozen": backends.flags_frozen,
        }

        self.assertIs(torch.backends, backends)
        self.assertIs(sys.modules["torch_rs.backends"], backends)
        self.assertIsInstance(backends, types.ModuleType)
        self.assertEqual(type(backends).__name__, "GenericModule")
        self.assertEqual(type(backends).__module__, "torch_rs.backends")
        self.assertIs(type(backends.m), types.ModuleType)
        self.assertIsNone(backends.__doc__)
        self.assertFalse(hasattr(backends, "__all__"))
        self.assertEqual(
            {name for name in vars(backends) if not name.startswith("_")},
            {*BACKEND_MODULES, "m"},
        )
        self.assertEqual(
            {name for name in vars(backends.m) if not name.startswith("_")},
            {
                *BACKEND_MODULES,
                "ContextProp",
                "GenericModule",
                "PropModule",
                *functions,
            },
        )
        self.assertIs(backends.GenericModule, type(backends))
        self.assertIs(
            backends.m.disable_global_flags,
            functions["disable_global_flags"],
        )
        self.assertIs(backends.m.flags_frozen, functions["flags_frozen"])

        direct_import = {}
        wildcard_import = {}
        exec(
            "from torch_rs.backends import disable_global_flags, flags_frozen",
            direct_import,
        )
        exec("from torch_rs.backends import *", wildcard_import)
        for name, function in functions.items():
            self.assertIs(direct_import[name], function)
            self.assertNotIn(name, wildcard_import)
        self.assertEqual(
            {name for name in wildcard_import if not name.startswith("__")},
            {*BACKEND_MODULES, "m"},
        )

        for name, function in functions.items():
            with self.subTest(function=name):
                self.assertIs(type(function), types.FunctionType)
                self.assertEqual(str(inspect.signature(function)), "()")
                self.assertEqual(inspect.get_annotations(function), {})
                self.assertEqual(function.__name__, name)
                self.assertEqual(function.__qualname__, name)
                self.assertEqual(function.__module__, "torch_rs.backends")
                self.assertIs(inspect.getmodule(function), backends)
                self.assertIsNone(function.__doc__)
                self.assertIsNone(function.__defaults__)
                self.assertIsNone(function.__kwdefaults__)
                self.assertEqual(function.__dict__, {})
                self.assertFalse(hasattr(function, "__text_signature__"))
                self.assertEqual(
                    function.__code__.co_names,
                    ("__allow_nonbracketed_mutation_flag",),
                )
                self.assertEqual(function.__code__.co_freevars, ())
                self.assertEqual(function.__code__.co_cellvars, ())
                self.assertIs(copy.copy(function), function)
                self.assertIs(copy.deepcopy(function), function)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    payload = pickle.dumps(function, protocol=protocol)
                    self.assertIn(b"torch_rs.backends", payload)
                    self.assertIs(pickle.loads(payload), function)

        cudnn = backends.cudnn
        for name in FLAG_NAMES:
            with self.subTest(descriptor=name):
                descriptor = vars(type(cudnn))[name]
                self.assertIs(type(descriptor), backends.ContextProp)
                self.assertEqual(set(vars(descriptor)), {"getter", "setter"})
                for copier in (copy.copy, copy.deepcopy):
                    copied = copier(descriptor)
                    self.assertIsNot(copied, descriptor)
                    self.assertIs(type(copied), backends.ContextProp)
                    self.assertIs(copied.getter, descriptor.getter)
                    self.assertIs(copied.setter, descriptor.setter)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    restored = pickle.loads(pickle.dumps(descriptor, protocol))
                    self.assertIs(type(restored), backends.ContextProp)
                    self.assertIs(restored.getter, descriptor.getter)
                    self.assertIs(restored.setter, descriptor.setter)

    def test_argument_errors_match_python_function_binding(self):
        for name in ("disable_global_flags", "flags_frozen"):
            function = getattr(torch.backends, name)
            cases = (
                (
                    lambda: function(None),
                    f"{name}() takes 0 positional arguments but 1 was given",
                ),
                (
                    lambda: function(None, None),
                    f"{name}() takes 0 positional arguments but 2 were given",
                ),
                (
                    lambda: function(enabled=True),
                    f"{name}() got an unexpected keyword argument 'enabled'",
                ),
            )
            for call, message in cases:
                with self.subTest(function=name, message=message):
                    with self.assertRaises(TypeError) as raised:
                        call()
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))

    def test_reload_matches_pytorch_module_replacement_behavior(self):
        self.run_script(
            f"""
import copy
import importlib
import pickle
import re
import sys

import torch_rs as torch

backends = torch.backends
cudnn = backends.cudnn
old_disable = backends.disable_global_flags
old_frozen = backends.flags_frozen
backends.disable_global_flags()
assert old_frozen() is True

reloaded_cudnn = importlib.reload(cudnn)
assert reloaded_cudnn is not cudnn
assert backends.cudnn is cudnn
assert sys.modules["torch_rs.backends.cudnn"] is reloaded_cudnn
assert reloaded_cudnn.m is cudnn
assert backends.flags_frozen() is True
for module in (cudnn, reloaded_cudnn):
    try:
        module.enabled = False
    except RuntimeError as error:
        assert str(error) == {FROZEN_ERROR!r}
    else:
        raise AssertionError("a reloaded cuDNN proxy unexpectedly became mutable")

reloaded = importlib.reload(backends)
assert reloaded is not backends
assert torch.backends is backends
assert sys.modules["torch_rs.backends"] is reloaded
assert importlib.import_module("torch_rs.backends") is reloaded
assert reloaded.m is backends
assert reloaded.flags_frozen() is False
assert backends.flags_frozen() is False
assert old_frozen() is True

try:
    cudnn.enabled = False
except RuntimeError as error:
    assert str(error) == {FROZEN_ERROR!r}
else:
    raise AssertionError("the stale cuDNN proxy unexpectedly became mutable")

for function in (old_disable, old_frozen):
    try:
        pickle.dumps(function)
    except pickle.PicklingError as error:
        message = re.sub(r"0x[0-9a-fA-F]+", "0x...", str(error))
        expected = (
            f"Can't pickle <function {{function.__name__}} at 0x...>: "
            "it's not the same object as "
            f"torch_rs.backends.{{function.__name__}}"
        )
        assert message == expected
    else:
        raise AssertionError("a stale backend function remained pickleable")

for function in (reloaded.disable_global_flags, reloaded.flags_frozen):
    assert copy.copy(function) is function
    assert copy.deepcopy(function) is function
    assert pickle.loads(pickle.dumps(function)) is function
"""
        )


if __name__ == "__main__":
    unittest.main()
