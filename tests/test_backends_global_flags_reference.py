import copy
import importlib
import inspect
import json
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


BEHAVIOR_SCRIPT = r'''
import importlib
import json
import sys
import threading

root = importlib.import_module(sys.argv[1])
package_name = root.__name__
backends = root.backends
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

def normalize(value):
    return str(value).replace(package_name, "torch")

def states(module):
    return tuple(getattr(module, name) for name in properties)

def capture_assignment(module, name, value):
    before = states(module)
    try:
        setattr(module, name, value)
    except BaseException as error:
        outcome = (
            type(error).__name__,
            normalize(error),
            tuple(normalize(argument) for argument in error.args),
        )
    else:
        outcome = None
    return outcome, before, states(module)

worker_ready = threading.Event()
main_disabled = threading.Event()
thread_observations = []
thread_errors = []

def worker():
    try:
        thread_observations.append(backends.flags_frozen())
        worker_ready.set()
        if not main_disabled.wait(timeout=10):
            raise RuntimeError("timed out waiting for main-thread transition")
        thread_observations.append(backends.flags_frozen())
        thread_observations.append(backends.disable_global_flags())
        thread_observations.append(backends.flags_frozen())
    except BaseException as error:
        thread_errors.append((type(error).__name__, normalize(error)))
        worker_ready.set()

original = cudnn.set_flags()
try:
    cudnn.set_flags(*default_state)
    initial = backends.flags_frozen()
    thread = threading.Thread(target=worker)
    thread.start()
    ready = worker_ready.wait(timeout=10)
    disable_result = backends.disable_global_flags()
    main_frozen = backends.flags_frozen()
    main_disabled.set()
    thread.join(timeout=10)

    assignments = [
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
        "initial": initial,
        "initial_type": type(initial).__name__,
        "ready": ready,
        "disable_result": disable_result,
        "main_frozen": main_frozen,
        "joined": not thread.is_alive(),
        "thread_observations": thread_observations,
        "thread_errors": thread_errors,
        "assignments": assignments,
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
    }))
finally:
    cudnn.set_flags(*original)
'''


RELOAD_SCRIPT = r'''
import importlib
import json
import pickle
import re
import sys

root = importlib.import_module(sys.argv[1])
package_name = root.__name__
backends = root.backends
cudnn = backends.cudnn
old_module = backends.m
old_disable = backends.disable_global_flags
old_frozen = backends.flags_frozen
namespace = backends.__dict__
backends.disable_global_flags()
reloaded = importlib.reload(backends)

def normalize(value):
    return str(value).replace(package_name, "torch")

def pickle_outcome(function):
    try:
        pickle.dumps(function)
    except BaseException as error:
        return (
            type(error).__name__,
            re.sub(r"0x[0-9a-fA-F]+", "0x...", normalize(error)),
        )
    return ("ok",)

before = cudnn.enabled
try:
    cudnn.enabled = not before
except BaseException as error:
    assignment = (
        type(error).__name__,
        normalize(error),
        tuple(normalize(argument) for argument in error.args),
    )
else:
    assignment = None

print(json.dumps({
    "reload_identity": (
        reloaded is not backends,
        backends.__dict__ is namespace,
        root.backends is backends,
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


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class BackendsGlobalFlagsReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "backends global flag differentials require pinned "
                "PyTorch 2.13.0"
            )
        if not reference_torch.backends.cudnn.is_available():
            raise unittest.SkipTest(
                "five-property cuDNN freezing differentials require a "
                "cuDNN-built reference PyTorch"
            )

    def run_contract(self, script, package_name):
        completed = subprocess.run(
            [sys.executable, "-c", script, package_name],
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

    def normalize(self, value):
        return str(value).replace("torch_rs", "torch")

    def capture_error(self, call):
        try:
            call()
        except BaseException as error:
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
                argument = self.normalize(argument)
            shape.append((opcode.name, argument))
        return shape

    def test_transition_cudnn_freezing_and_child_reload_match(self):
        self.assertEqual(
            self.run_contract(BEHAVIOR_SCRIPT, "torch_rs"),
            self.run_contract(BEHAVIOR_SCRIPT, "torch"),
        )

    def test_parent_reload_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.run_contract(RELOAD_SCRIPT, "torch_rs"),
            self.run_contract(RELOAD_SCRIPT, "torch"),
        )

    def test_metadata_imports_wildcards_copying_and_pickling_match(self):
        actual_backends = torch.backends
        expected_backends = reference_torch.backends
        supported_wildcard = {*BACKEND_NAMES, "m"}

        self.assertEqual(
            type(actual_backends).__name__,
            type(expected_backends).__name__,
        )
        self.assertEqual(
            self.normalize(type(actual_backends).__module__),
            type(expected_backends).__module__,
        )
        self.assertEqual(actual_backends.__doc__, expected_backends.__doc__)
        self.assertEqual(
            hasattr(actual_backends, "__all__"),
            hasattr(expected_backends, "__all__"),
        )
        self.assertEqual(
            {
                name
                for name in vars(actual_backends)
                if not name.startswith("_")
            },
            supported_wildcard,
        )
        self.assertEqual(
            {
                name
                for name in vars(expected_backends)
                if name in supported_wildcard
            },
            supported_wildcard,
        )

        actual_wildcard = {}
        expected_wildcard = {}
        exec("from torch_rs.backends import *", actual_wildcard)
        exec("from torch.backends import *", expected_wildcard)
        self.assertEqual(
            {
                name
                for name in actual_wildcard
                if not name.startswith("__")
            },
            {
                name
                for name in expected_wildcard
                if name in supported_wildcard
            },
        )

        for name in ("disable_global_flags", "flags_frozen"):
            actual = getattr(actual_backends, name)
            expected = getattr(expected_backends, name)
            actual_import = {}
            expected_import = {}
            exec(f"from torch_rs.backends import {name}", actual_import)
            exec(f"from torch.backends import {name}", expected_import)

            with self.subTest(function=name):
                self.assertNotIn(name, actual_wildcard)
                self.assertNotIn(name, expected_wildcard)
                self.assertIs(actual_import[name], actual)
                self.assertIs(expected_import[name], expected)
                self.assertIs(actual, getattr(actual_backends.m, name))
                self.assertIs(expected, getattr(expected_backends.m, name))
                self.assertIs(type(actual), types.FunctionType)
                self.assertIs(type(expected), types.FunctionType)
                self.assertEqual(
                    str(inspect.signature(actual)),
                    str(inspect.signature(expected)),
                )
                self.assertEqual(
                    inspect.get_annotations(actual),
                    inspect.get_annotations(expected),
                )
                self.assertEqual(actual.__name__, expected.__name__)
                self.assertEqual(actual.__qualname__, expected.__qualname__)
                self.assertEqual(
                    self.normalize(actual.__module__),
                    expected.__module__,
                )
                self.assertIs(inspect.getmodule(actual), actual_backends)
                self.assertIs(inspect.getmodule(expected), expected_backends)
                self.assertEqual(actual.__doc__, expected.__doc__)
                self.assertEqual(actual.__defaults__, expected.__defaults__)
                self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
                self.assertEqual(actual.__dict__, expected.__dict__)
                self.assertEqual(
                    hasattr(actual, "__text_signature__"),
                    hasattr(expected, "__text_signature__"),
                )
                self.assertEqual(
                    actual.__code__.co_names,
                    expected.__code__.co_names,
                )
                self.assertEqual(
                    actual.__code__.co_freevars,
                    expected.__code__.co_freevars,
                )
                self.assertEqual(
                    actual.__code__.co_cellvars,
                    expected.__code__.co_cellvars,
                )
                self.assertIs(copy.copy(actual), actual)
                self.assertIs(copy.deepcopy(actual), actual)
                self.assertIs(copy.copy(expected), expected)
                self.assertIs(copy.deepcopy(expected), expected)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    self.assertIs(
                        pickle.loads(pickle.dumps(actual, protocol)),
                        actual,
                    )
                    self.assertIs(
                        pickle.loads(pickle.dumps(expected, protocol)),
                        expected,
                    )
                    self.assertEqual(
                        self.pickle_shape(actual, protocol),
                        self.pickle_shape(expected, protocol),
                    )

        for actual_call, expected_call in (
            (
                lambda: actual_backends.disable_global_flags(None),
                lambda: expected_backends.disable_global_flags(None),
            ),
            (
                lambda: actual_backends.disable_global_flags(enabled=True),
                lambda: expected_backends.disable_global_flags(enabled=True),
            ),
            (
                lambda: actual_backends.flags_frozen(None),
                lambda: expected_backends.flags_frozen(None),
            ),
            (
                lambda: actual_backends.flags_frozen(enabled=True),
                lambda: expected_backends.flags_frozen(enabled=True),
            ),
        ):
            self.assertEqual(
                self.capture_error(actual_call),
                self.capture_error(expected_call),
            )

        for name in ("ContextProp", "PropModule", "GenericModule"):
            actual = getattr(actual_backends, name)
            expected = getattr(expected_backends, name)
            self.assertEqual(actual.__name__, expected.__name__)
            self.assertEqual(actual.__qualname__, expected.__qualname__)
            self.assertEqual(self.normalize(actual.__module__), expected.__module__)
            self.assertIs(copy.copy(actual), actual)
            self.assertIs(copy.deepcopy(actual), actual)
            self.assertIs(pickle.loads(pickle.dumps(actual)), actual)

        actual_cudnn = actual_backends.cudnn
        expected_cudnn = expected_backends.cudnn
        self.assertIs(actual_cudnn.__class__.__mro__[1], actual_backends.PropModule)
        self.assertIs(
            expected_cudnn.__class__.__mro__[1],
            expected_backends.PropModule,
        )
        for name in PROPERTIES:
            actual_descriptor = vars(type(actual_cudnn))[name]
            expected_descriptor = vars(type(expected_cudnn))[name]
            self.assertIs(type(actual_descriptor), actual_backends.ContextProp)
            self.assertIs(type(expected_descriptor), expected_backends.ContextProp)
            self.assertEqual(
                set(actual_descriptor.__dict__),
                set(expected_descriptor.__dict__),
            )


if __name__ == "__main__":
    unittest.main()
