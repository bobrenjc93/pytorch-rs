import copy
import importlib
import inspect
import pickle
import subprocess
import sys
import threading
import types
import unittest

import torch_rs as torch


FUNCTION_DOC = """Check for __torch_function__ implementations in the elements of an iterable
    or if a __torch_function__ mode is enabled.  Considers exact ``Tensor`` s
    and ``Parameter`` s non-dispatchable.  Use this to guard a call to
    :func:`handle_torch_function`; don't use it to test if something
    is Tensor-like, use :func:`is_tensor_like` instead.
    Arguments
    ---------
    relevant_args : iterable
        Iterable or arguments to check for __torch_function__ methods.
    Returns
    -------
    bool
        True if any of the elements of relevant_args have __torch_function__
        implementations, False otherwise.
    See Also
    ________
    torch.is_tensor_like
        Checks if something is a Tensor-like, including an exact ``Tensor``.
    """


class HasTorchFunctionTests(unittest.TestCase):
    def test_exact_tensors_and_custom_overrides(self):
        function = torch.overrides.has_torch_function
        exact = torch.tensor([1.0])

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        class NoneOverride:
            __torch_function__ = None

        class RaisingDescriptor:
            def __init__(self):
                self.lookups = 0

            def __get__(self, instance, owner):
                self.lookups += 1
                raise RuntimeError("descriptor boom")

        descriptor = RaisingDescriptor()

        class BrokenOverride:
            __torch_function__ = descriptor

        cases = (
            ((), False),
            ([], False),
            ((exact,), False),
            ((exact, object()), False),
            ((torch.Tensor,), True),
            ((Override(),), True),
            ((object(), Override()), True),
            ((NoneOverride(),), True),
            ((BrokenOverride(),), False),
            ((object(),), False),
            ("text", False),
        )
        for relevant_args, expected in cases:
            with self.subTest(relevant_args=type(relevant_args).__name__):
                result = function(relevant_args)
                self.assertIs(type(result), bool)
                self.assertIs(result, expected)
        self.assertEqual(descriptor.lookups, 1)

    def test_arbitrary_iterables_are_fully_materialized(self):
        function = torch.overrides.has_torch_function

        class Override:
            __torch_function__ = None

        class RecordingIterable:
            def __init__(self, values):
                self.values = values
                self.yielded = []

            def __iter__(self):
                for value in self.values:
                    self.yielded.append(value)
                    yield value

        override = Override()
        trailing = object()
        iterable = RecordingIterable([override, trailing])
        self.assertIs(function(iterable), True)
        self.assertEqual(iterable.yielded, [override, trailing])

        generator = (value for value in [object(), override])
        self.assertIs(function(generator), True)
        self.assertEqual(list(generator), [])
        self.assertIs(function({override: "value"}), True)
        self.assertIs(function([[override]]), False)

        class LateFailure:
            def __iter__(self):
                yield override
                raise RuntimeError("late iteration failure")

        with self.assertRaisesRegex(RuntimeError, "^late iteration failure$"):
            function(LateFailure())

    def test_active_modes_preserve_the_complete_stack(self):
        function = torch.overrides.has_torch_function
        tensor = torch.tensor([1.0])

        class Mode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                return NotImplemented

        lower = Mode("lower")
        upper = Mode("upper")
        self.assertIs(function((tensor,)), False)
        self.assertIs(function((object(),)), False)

        with lower:
            self.assertEqual(
                torch.overrides._get_current_function_mode_stack(), [lower]
            )
            self.assertIs(function(()), False)
            self.assertIs(function((tensor,)), True)
            self.assertIs(function((object(),)), True)
            self.assertEqual(
                torch.overrides._get_current_function_mode_stack(), [lower]
            )
            with upper:
                before = torch.overrides._get_current_function_mode_stack()
                self.assertEqual(before, [lower, upper])
                self.assertIs(function((tensor,)), True)
                self.assertEqual(
                    torch.overrides._get_current_function_mode_stack(), before
                )

                class BrokenIterable:
                    def __iter__(self):
                        raise RuntimeError("iteration boom")

                with self.assertRaisesRegex(RuntimeError, "^iteration boom$"):
                    function(BrokenIterable())
                self.assertEqual(
                    torch.overrides._get_current_function_mode_stack(), before
                )
            self.assertEqual(
                torch.overrides._get_current_function_mode_stack(), [lower]
            )
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])
        self.assertIs(function((tensor,)), False)

    def test_mode_stack_is_thread_local_and_unchanged_by_queries(self):
        function = torch.overrides.has_torch_function
        tensor = torch.tensor([1.0])
        worker_entered = threading.Event()
        worker_leave = threading.Event()
        worker_results = []
        errors = []

        class Mode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                return NotImplemented

        main_mode = Mode("main")

        def worker():
            worker_mode = Mode("worker")
            try:
                worker_results.append(
                    (
                        function((tensor,)),
                        torch.overrides._get_current_function_mode_stack(),
                    )
                )
                with worker_mode:
                    worker_results.append(
                        (
                            function((tensor,)),
                            torch.overrides._get_current_function_mode_stack(),
                        )
                    )
                    worker_entered.set()
                    if not worker_leave.wait(timeout=10):
                        raise RuntimeError("main thread did not release worker")
                worker_results.append(
                    (
                        function((tensor,)),
                        torch.overrides._get_current_function_mode_stack(),
                    )
                )
            except BaseException as error:
                errors.append(error)
                worker_entered.set()

        thread = threading.Thread(target=worker)
        with main_mode:
            thread.start()
            self.assertTrue(worker_entered.wait(timeout=10))
            self.assertEqual(errors, [])
            self.assertIs(function((tensor,)), True)
            self.assertEqual(
                torch.overrides._get_current_function_mode_stack(), [main_mode]
            )
            worker_leave.set()
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(worker_results), 3)
        self.assertIs(worker_results[0][0], False)
        self.assertEqual(worker_results[0][1], [])
        self.assertIs(worker_results[1][0], True)
        self.assertEqual(len(worker_results[1][1]), 1)
        self.assertEqual(worker_results[1][1][0].label, "worker")
        self.assertIs(worker_results[2][0], False)
        self.assertEqual(worker_results[2][1], [])
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

    def test_native_metadata_aliases_and_exports(self):
        overrides = importlib.import_module("torch_rs.overrides")
        function = overrides.has_torch_function

        self.assertIs(torch.overrides, overrides)
        self.assertIs(function, overrides._has_torch_function)
        self.assertIs(function, torch._C._has_torch_function)
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "_has_torch_function")
        self.assertEqual(function.__qualname__, "_has_torch_function")
        self.assertEqual(function.__module__, "torch_rs._C")
        self.assertIs(function.__self__, torch._C)
        self.assertIs(inspect.getmodule(function), torch._C)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
        with self.assertRaisesRegex(ValueError, "^no signature found for builtin"):
            inspect.signature(function)

        self.assertEqual(overrides.__all__, ["TorchFunctionMode", "has_torch_function"])
        namespace = {}
        exec("from torch_rs.overrides import *", namespace)
        self.assertIs(namespace["TorchFunctionMode"], overrides.TorchFunctionMode)
        self.assertIs(namespace["has_torch_function"], function)
        self.assertNotIn("_has_torch_function", namespace)
        self.assertNotIn("_has_torch_function", torch._C.__all__)
        self.assertFalse(hasattr(torch, "has_torch_function"))
        self.assertNotIn("has_torch_function", torch.__all__)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(function, protocol)), function)

    def test_argument_and_iteration_errors_match_the_native_contract(self):
        function = torch.overrides.has_torch_function
        cases = (
            (
                lambda: function(),
                "torch_rs._C._has_torch_function() takes exactly one argument (0 given)",
            ),
            (
                lambda: function((), ()),
                "torch_rs._C._has_torch_function() takes exactly one argument (2 given)",
            ),
            (
                lambda: function(relevant_args=()),
                "torch_rs._C._has_torch_function() takes no keyword arguments",
            ),
            (
                lambda: function(42),
                "expected a sequence",
            ),
            (
                lambda: function(None),
                "expected a sequence",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

        class BrokenIterable:
            def __iter__(self):
                raise ValueError("custom iteration failure")

        with self.assertRaisesRegex(ValueError, "^custom iteration failure$"):
            function(BrokenIterable())

    def test_importing_and_calling_does_not_import_pytorch(self):
        script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

tensor = torch.tensor([1.0])
assert torch.overrides.has_torch_function((tensor,)) is False

class Override:
    __torch_function__ = None

assert torch.overrides.has_torch_function(iter([tensor, Override()])) is True

class Mode(torch.overrides.TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        return NotImplemented

mode = Mode()
with mode:
    assert torch.overrides.has_torch_function((tensor,)) is True
    assert torch.overrides._get_current_function_mode_stack() == [mode]
assert torch.overrides._get_current_function_mode_stack() == []
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
"""
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


if __name__ == "__main__":
    unittest.main()
