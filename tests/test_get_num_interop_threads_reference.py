import importlib.util
import subprocess
import sys
import textwrap
import unittest


REFERENCE_AVAILABLE = importlib.util.find_spec("torch") is not None
REFERENCE_PREAMBLE = r"""
import torch as reference_torch

if reference_torch.__version__.split("+")[0] != "2.13.0":
    raise AssertionError(
        "get_num_interop_threads differentials require pinned PyTorch 2.13.0"
    )

# PyTorch permits this process-global setting only before inter-op work starts.
# Configure each fresh reference process before importing the implementation.
reference_torch.set_num_interop_threads(1)
assert reference_torch.get_num_interop_threads() == 1

import torch_rs as torch
"""


@unittest.skipUnless(REFERENCE_AVAILABLE, "install the reference dependency group")
class GetNumInteropThreadsReferenceTests(unittest.TestCase):
    def run_reference_process(self, body):
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                REFERENCE_PREAMBLE + "\n" + textwrap.dedent(body),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )

    def test_single_worker_grad_and_thread_states_match_pytorch_2_13(self):
        self.run_reference_process(
            r"""
            import contextlib
            import threading

            def threaded_outcome(module):
                function = module.get_num_interop_threads

                def query():
                    before = module.is_grad_enabled()
                    first = function()
                    middle = module.is_grad_enabled()
                    second = function()
                    after = module.is_grad_enabled()
                    return (
                        before,
                        type(first) is int,
                        first,
                        middle,
                        type(second) is int,
                        second,
                        after,
                    )

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
                        context = (
                            module.no_grad()
                            if index % 2
                            else contextlib.nullcontext()
                        )
                        with context:
                            barrier.wait(timeout=10)
                            worker_states[index] = query()
                    except BaseException as error:
                        errors.append((type(error).__name__, str(error)))

                threads = [
                    threading.Thread(target=worker, args=(index,))
                    for index in range(worker_count)
                ]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=10)

                assert not any(thread.is_alive() for thread in threads)
                assert errors == []
                return main_states, worker_states

            assert threaded_outcome(torch) == threaded_outcome(reference_torch)
            assert torch.get_num_interop_threads() == 1
            assert reference_torch.get_num_interop_threads() == 1
            """
        )

    def test_builtin_contract_matches_pytorch_2_13(self):
        self.run_reference_process(
            r"""
            import inspect
            import pickle
            import types

            actual = torch.get_num_interop_threads
            expected = reference_torch.get_num_interop_threads

            assert type(actual) is types.BuiltinFunctionType
            assert type(expected) is types.BuiltinFunctionType
            assert actual.__name__ == expected.__name__
            assert actual.__qualname__ == expected.__qualname__
            assert (
                actual.__module__.replace("torch_rs.torch_rs", "torch")
                == expected.__module__
            )
            assert actual.__doc__ == expected.__doc__
            assert actual.__text_signature__ == expected.__text_signature__
            assert repr(actual) == repr(expected)
            assert actual.__self__ is torch._C
            assert expected.__self__ is reference_torch._C
            assert torch._C.get_num_interop_threads is actual
            assert reference_torch._C.get_num_interop_threads is expected

            def signature_outcome(function):
                try:
                    return "value", str(inspect.signature(function))
                except Exception as error:
                    return "error", type(error).__name__, str(error)

            assert signature_outcome(actual) == signature_outcome(expected)
            assert torch.__all__.count("get_num_interop_threads") == 1
            assert (
                torch.__all__.count("get_num_interop_threads")
                == reference_torch.__all__.count("get_num_interop_threads")
            )

            for module, function in (
                (torch, actual),
                (reference_torch, expected),
            ):
                wildcard_namespace = {}
                exec(f"from {module.__name__} import *", wildcard_namespace)
                assert wildcard_namespace["get_num_interop_threads"] is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    restored = pickle.loads(pickle.dumps(function, protocol=protocol))
                    assert restored is function
            """
        )

    def test_no_argument_errors_match_pytorch_2_13(self):
        self.run_reference_process(
            r"""
            def error_outcome(call):
                try:
                    call()
                except Exception as error:
                    return type(error), str(error), error.args
                raise AssertionError("call unexpectedly succeeded")

            cases = (
                (
                    lambda: torch.get_num_interop_threads(None),
                    lambda: reference_torch.get_num_interop_threads(None),
                ),
                (
                    lambda: torch.get_num_interop_threads(None, None),
                    lambda: reference_torch.get_num_interop_threads(None, None),
                ),
                (
                    lambda: torch.get_num_interop_threads(threads=None),
                    lambda: reference_torch.get_num_interop_threads(threads=None),
                ),
                (
                    lambda: torch.get_num_interop_threads(None, threads=None),
                    lambda: reference_torch.get_num_interop_threads(
                        None, threads=None
                    ),
                ),
            )
            for actual_call, expected_call in cases:
                assert error_outcome(actual_call) == error_outcome(expected_call)

            assert torch.get_num_interop_threads(**{}) == 1
            assert reference_torch.get_num_interop_threads(**{}) == 1
            """
        )

    def test_setters_stay_absent_and_intraop_behavior_is_unchanged(self):
        self.run_reference_process(
            r"""
            assert torch.get_num_threads() == 1
            for _ in range(4):
                assert torch.get_num_interop_threads() == 1
                assert torch.get_num_threads() == 1

            for name in ("set_num_threads", "set_num_interop_threads"):
                assert not hasattr(torch, name)
                assert not hasattr(torch._C, name)
                assert name not in torch.__all__
                assert hasattr(reference_torch, name)
                assert hasattr(reference_torch._C, name)
                assert name in reference_torch.__all__
            """
        )


if __name__ == "__main__":
    unittest.main()
