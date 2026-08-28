import copy
import importlib
import inspect
import json
import pickle
import re
import subprocess
import sys
import types
import unittest

import torch_rs as torch


BACKEND_NAMES = {
    "cpu",
    "cuda",
    "cudnn",
    "kleidiai",
    "mha",
    "mkl",
    "nnpack",
    "openmp",
}
PROPERTIES = (
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
        return json.loads(completed.stdout)

    def test_initial_state_one_way_transition_and_thread_visibility(self):
        result = self.run_script(
            r'''
import json
import threading

import torch_rs as torch

backends = torch.backends
worker_ready = threading.Event()
main_disabled = threading.Event()
observations = []
errors = []

def worker():
    try:
        observations.append(("worker-initial", backends.flags_frozen()))
        worker_ready.set()
        if not main_disabled.wait(timeout=10):
            raise RuntimeError("timed out waiting for main-thread transition")
        observations.append(("worker-after-main", backends.flags_frozen()))
        observations.append(("worker-disable", backends.disable_global_flags()))
        observations.append(("worker-final", backends.flags_frozen()))
    except BaseException as error:
        errors.append((type(error).__name__, str(error)))
        worker_ready.set()

initial = backends.flags_frozen()
thread = threading.Thread(target=worker)
thread.start()
ready = worker_ready.wait(timeout=10)
main_result = backends.disable_global_flags()
main_after = backends.flags_frozen()
main_disabled.set()
thread.join(timeout=10)
repeat_result = backends.disable_global_flags()

print(json.dumps({
    "initial": initial,
    "initial_type": type(initial).__name__,
    "ready": ready,
    "main_result": main_result,
    "main_after": main_after,
    "repeat_result": repeat_result,
    "final": backends.flags_frozen(),
    "joined": not thread.is_alive(),
    "observations": observations,
    "errors": errors,
}))
'''
        )
        self.assertEqual(
            result,
            {
                "initial": False,
                "initial_type": "bool",
                "ready": True,
                "main_result": None,
                "main_after": True,
                "repeat_result": None,
                "final": True,
                "joined": True,
                "observations": [
                    ["worker-initial", False],
                    ["worker-after-main", True],
                    ["worker-disable", None],
                    ["worker-final", True],
                ],
                "errors": [],
            },
        )

    def test_frozen_cudnn_properties_and_bracketed_mutation(self):
        result = self.run_script(
            r'''
import importlib
import json
import sys

import torch_rs as torch

backends = torch.backends
cudnn = backends.cudnn
properties = (
    "enabled",
    "benchmark",
    "benchmark_limit",
    "deterministic",
    "allow_tf32",
)
default_state = (True, False, 10, False, True)
target_state = (False, True, 17, True, False)

def states(module):
    return tuple(getattr(module, name) for name in properties)

def capture_assignment(module, name, value):
    before = states(module)
    try:
        setattr(module, name, value)
    except BaseException as error:
        outcome = (type(error).__name__, str(error), error.args)
    else:
        outcome = None
    return outcome, before, states(module)

original = cudnn.set_flags()
try:
    cudnn.set_flags(*default_state)
    backends.disable_global_flags()
    assignment_results = [
        (name, *capture_assignment(cudnn, name, value))
        for name, value in zip(properties, target_state)
    ]

    set_flags_result = cudnn.set_flags(*target_state)
    after_set_flags = states(cudnn)
    with cudnn.flags(*default_state) as entered:
        context_state = states(cudnn)
        context_frozen = backends.flags_frozen()
        context_assignment = capture_assignment(cudnn, "enabled", False)
    after_context = states(cudnn)

    namespace = cudnn.__dict__
    reloaded = importlib.reload(cudnn)
    reload_identity = (
        reloaded is not cudnn,
        cudnn.__dict__ is namespace,
        backends.cudnn is cudnn,
        sys.modules[cudnn.__name__] is reloaded,
        reloaded.m is cudnn,
    )
    old_reload_assignment = capture_assignment(cudnn, "benchmark", False)
    new_reload_assignment = capture_assignment(reloaded, "benchmark", False)

    print(json.dumps({
        "frozen": backends.flags_frozen(),
        "assignments": assignment_results,
        "set_flags_result": set_flags_result,
        "after_set_flags": after_set_flags,
        "entered": entered,
        "context_state": context_state,
        "context_frozen": context_frozen,
        "context_assignment": context_assignment,
        "after_context": after_context,
        "reload_identity": reload_identity,
        "old_reload_assignment": old_reload_assignment,
        "new_reload_assignment": new_reload_assignment,
        "available": cudnn.is_available(),
        "version": cudnn.version(),
        "execution": hasattr(torch, "cudnn_convolution"),
    }))
finally:
    cudnn.set_flags(*original)
'''
        )
        expected_error = ["RuntimeError", FROZEN_ERROR, [FROZEN_ERROR]]
        expected_assignment = [expected_error, list(DEFAULT_STATE), list(DEFAULT_STATE)]
        self.assertEqual(result["frozen"], True)
        self.assertEqual(
            result["assignments"],
            [
                [name, *expected_assignment]
                for name in PROPERTIES
            ],
        )
        self.assertEqual(
            result["set_flags_result"],
            [*DEFAULT_STATE, "none", "auto"],
        )
        self.assertEqual(result["after_set_flags"], list(TARGET_STATE))
        self.assertIsNone(result["entered"])
        self.assertEqual(result["context_state"], list(DEFAULT_STATE))
        self.assertEqual(result["context_frozen"], True)
        self.assertEqual(
            result["context_assignment"],
            [expected_error, list(DEFAULT_STATE), list(DEFAULT_STATE)],
        )
        self.assertEqual(result["after_context"], list(TARGET_STATE))
        self.assertEqual(result["reload_identity"], [True] * 5)
        reloaded_expected = [
            expected_error,
            list(TARGET_STATE),
            list(TARGET_STATE),
        ]
        self.assertEqual(result["old_reload_assignment"], reloaded_expected)
        self.assertEqual(result["new_reload_assignment"], reloaded_expected)
        self.assertIs(result["available"], False)
        self.assertIsNone(result["version"])
        self.assertIs(result["execution"], False)

    def test_callable_metadata_imports_wildcards_copying_and_pickling(self):
        backends = importlib.import_module("torch_rs.backends")
        functions = {
            "disable_global_flags": backends.disable_global_flags,
            "flags_frozen": backends.flags_frozen,
        }

        self.assertIs(torch.backends, backends)
        self.assertIs(sys.modules["torch_rs.backends"], backends)
        self.assertEqual(type(backends).__name__, "GenericModule")
        self.assertEqual(type(backends).__module__, "torch_rs.backends")
        self.assertIsNone(backends.__doc__)
        self.assertFalse(hasattr(backends, "__all__"))
        self.assertEqual(
            {name for name in vars(backends) if not name.startswith("_")},
            {*BACKEND_NAMES, "m"},
        )
        self.assertTrue(
            {
                *BACKEND_NAMES,
                "ContextProp",
                "GenericModule",
                "PropModule",
                "contextmanager",
                "disable_global_flags",
                "flags_frozen",
                "sys",
                "torch",
                "types",
            }.issubset(
                {
                    name
                    for name in vars(backends.m)
                    if not name.startswith("_")
                }
            )
        )

        direct_imports = {}
        wildcard = {}
        exec(
            "from torch_rs.backends import disable_global_flags, flags_frozen",
            direct_imports,
        )
        exec("from torch_rs.backends import *", wildcard)
        for name, function in functions.items():
            self.assertIs(direct_imports[name], function)
            self.assertIs(getattr(backends.m, name), function)
            self.assertNotIn(name, wildcard)
        self.assertEqual(
            {name for name in wildcard if not name.startswith("__")},
            {*BACKEND_NAMES, "m"},
        )
        self.assertIs(wildcard["m"], backends.m)

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

        for call, message in (
            (
                lambda: functions["disable_global_flags"](None),
                "disable_global_flags() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: functions["disable_global_flags"](enabled=True),
                "disable_global_flags() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: functions["flags_frozen"](None),
                "flags_frozen() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: functions["flags_frozen"](enabled=True),
                "flags_frozen() got an unexpected keyword argument 'enabled'",
            ),
        ):
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

        cudnn = backends.cudnn
        self.assertIs(cudnn.__class__.__mro__[1], backends.PropModule)
        for name in PROPERTIES:
            descriptor = vars(type(cudnn))[name]
            self.assertIs(type(descriptor), backends.ContextProp)

    def test_parent_reload_matches_pytorch_proxy_state_split(self):
        result = self.run_script(
            r'''
import importlib
import json
import pickle
import re
import sys

import torch_rs as torch

backends = torch.backends
cudnn = backends.cudnn
old_module = backends.m
old_disable = backends.disable_global_flags
old_frozen = backends.flags_frozen
namespace = backends.__dict__
backends.disable_global_flags()
reloaded = importlib.reload(backends)

def pickle_outcome(function):
    try:
        pickle.dumps(function)
    except BaseException as error:
        return (
            type(error).__name__,
            re.sub(r"0x[0-9a-fA-F]+", "0x...", str(error)),
        )
    return ("ok",)

before = cudnn.enabled
try:
    cudnn.enabled = not before
except BaseException as error:
    assignment = (type(error).__name__, str(error), error.args)
else:
    assignment = None

print(json.dumps({
    "reload_identity": (
        reloaded is not backends,
        backends.__dict__ is namespace,
        torch.backends is backends,
        sys.modules[backends.__name__] is reloaded,
        reloaded.m is backends,
        backends.m is old_module,
    ),
    "queries": (
        old_frozen(),
        backends.flags_frozen(),
        reloaded.flags_frozen(),
    ),
    "functions_replaced": (
        backends.disable_global_flags is not old_disable,
        backends.flags_frozen is not old_frozen,
        reloaded.disable_global_flags is backends.disable_global_flags,
        reloaded.flags_frozen is backends.flags_frozen,
    ),
    "old_disable_pickle": pickle_outcome(old_disable),
    "old_frozen_pickle": pickle_outcome(old_frozen),
    "new_disable_pickle": pickle_outcome(backends.disable_global_flags),
    "new_frozen_pickle": pickle_outcome(backends.flags_frozen),
    "assignment": assignment,
    "state_unchanged": cudnn.enabled is before,
}))
'''
        )
        self.assertEqual(result["reload_identity"], [True] * 6)
        self.assertEqual(result["queries"], [True, False, False])
        self.assertEqual(result["functions_replaced"], [True] * 4)
        self.assertEqual(
            result["old_disable_pickle"],
            [
                "PicklingError",
                "Can't pickle <function disable_global_flags at 0x...>: "
                "it's not the same object as "
                "torch_rs.backends.disable_global_flags",
            ],
        )
        self.assertEqual(
            result["old_frozen_pickle"],
            [
                "PicklingError",
                "Can't pickle <function flags_frozen at 0x...>: "
                "it's not the same object as torch_rs.backends.flags_frozen",
            ],
        )
        self.assertEqual(result["new_disable_pickle"], ["ok"])
        self.assertEqual(result["new_frozen_pickle"], ["ok"])
        self.assertEqual(
            result["assignment"],
            ["RuntimeError", FROZEN_ERROR, [FROZEN_ERROR]],
        )
        self.assertIs(result["state_unchanged"], True)


if __name__ == "__main__":
    unittest.main()
