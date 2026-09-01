import importlib.util
import json
import subprocess
import sys
import unittest

import torch_rs as torch

if __package__:
    from .signature_utils import assert_no_argument_signature
else:
    from signature_utils import assert_no_argument_signature


REFERENCE_AVAILABLE = importlib.util.find_spec("torch") is not None

CONTRACT_SCRIPT = r"""
import contextlib
import importlib
import inspect
import json
import pickle
import sys
import threading
import types

module = importlib.import_module(sys.argv[1])
if module.__name__ == "torch":
    if module.__version__.split("+")[0] != "2.13.0":
        raise AssertionError(
            "get_num_interop_threads differentials require pinned PyTorch 2.13.0"
        )

    # The setter is process-global and may only be used before parallel work
    # starts. Configure it in this disposable reference process so every other
    # PyTorch differential remains untouched.
    module.set_num_interop_threads(1)


def signature_outcome(function):
    try:
        return ["return", str(inspect.signature(function))]
    except Exception as error:
        return ["raise", type(error).__name__, str(error), list(error.args)]


def error_outcome(call):
    try:
        call()
    except Exception as error:
        return [type(error).__name__, str(error), list(error.args)]
    return ["return"]


def threaded_outcome():
    function = module.get_num_interop_threads

    def query():
        before = module.is_grad_enabled()
        first = function()
        middle = module.is_grad_enabled()
        second = function()
        after = module.is_grad_enabled()
        return [
            before,
            type(first) is int,
            first,
            middle,
            type(second) is int,
            second,
            after,
        ]

    main_states = [query()]
    with module.no_grad():
        main_states.append(query())
        with module.no_grad():
            main_states.append(query())
        main_states.append(query())
    main_states.append(query())

    worker_count = 8
    barrier = threading.Barrier(worker_count)
    worker_states = [None] * worker_count
    errors = []

    def worker(index):
        try:
            context = module.no_grad() if index % 2 else contextlib.nullcontext()
            with context:
                barrier.wait(timeout=10)
                worker_states[index] = query()
        except BaseException as error:
            errors.append([type(error).__name__, str(error)])

    threads = [
        threading.Thread(target=worker, args=(index,))
        for index in range(worker_count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    return {
        "main": main_states,
        "workers": worker_states,
        "errors": errors,
        "alive": [thread.is_alive() for thread in threads],
    }


function = module.get_num_interop_threads
wildcard_namespace = {}
exec(f"from {module.__name__} import *", wildcard_namespace)
print(
    json.dumps(
        {
            "value": [type(function()) is int, function()],
            "threaded": threaded_outcome(),
            "metadata": {
                "type_is_builtin": type(function) is types.BuiltinFunctionType,
                "name": function.__name__,
                "qualname": function.__qualname__,
                "module": function.__module__.replace("torch_rs.torch_rs", "torch"),
                "doc": function.__doc__,
                "text_signature": function.__text_signature__,
                "signature": signature_outcome(function),
                "repr": repr(function),
                "self_is_c": function.__self__ is module._C,
                "c_identity": module._C.get_num_interop_threads is function,
                "all_count": module.__all__.count("get_num_interop_threads"),
                "wildcard_identity": (
                    wildcard_namespace["get_num_interop_threads"] is function
                ),
                "pickle_identity": [
                    pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                    for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
                ],
            },
            "errors": [
                error_outcome(lambda: function(None)),
                error_outcome(lambda: function(None, None)),
                error_outcome(lambda: function(threads=None)),
                error_outcome(lambda: function(None, threads=None)),
            ],
            "setters": {
                name: [
                    hasattr(module, name),
                    hasattr(module._C, name),
                    module.__all__.count(name),
                ]
                for name in ("set_num_threads", "set_num_interop_threads")
            },
        }
    )
)
"""


@unittest.skipUnless(REFERENCE_AVAILABLE, "install the reference dependency group")
class GetNumInteropThreadsReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def contract(module_name):
            completed = subprocess.run(
                [sys.executable, "-I", "-c", CONTRACT_SCRIPT, module_name],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise AssertionError(completed.stdout + completed.stderr)
            return json.loads(completed.stdout)

        cls.actual = contract("torch_rs")
        cls.reference = contract("torch")

    def test_single_interop_worker_grad_and_thread_states_match_pytorch_2_13(self):
        self.assertEqual(self.actual["value"], self.reference["value"])
        self.assertEqual(self.actual["threaded"], self.reference["threaded"])

    def test_builtin_contract_matches_pytorch_2_13(self):
        self.assertEqual(self.actual["metadata"], self.reference["metadata"])
        assert_no_argument_signature(self, torch.get_num_interop_threads, "()")

    def test_no_argument_errors_match_pytorch_2_13(self):
        self.assertEqual(self.actual["errors"], self.reference["errors"])

    def test_thread_setters_are_exported_like_pytorch_2_13(self):
        for name in ("set_num_threads", "set_num_interop_threads"):
            with self.subTest(name=name):
                self.assertEqual(self.reference["setters"][name], [True, True, 1])
                self.assertEqual(self.actual["setters"][name], [True, True, 1])

        self.assertIs(torch._C.get_num_threads, torch.get_num_threads)
        self.assertIs(torch.get_num_threads(), 1)


if __name__ == "__main__":
    unittest.main()
