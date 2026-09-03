import contextlib
import copy
import importlib
import inspect
import pickle
import subprocess
import sys
import types
import typing
import unittest

import torch_rs as torch


FUNCTION_DOC = """
    A function for skipping all guards on a compiled function.

    WARNING: This function will drop all the safety guarantees from Dynamo
             compiled function. Use this with caution.

    To use this API, use guard_filter_fn argument while calling torch.compile

    >> opt_mod = torch.compile(
    >>     mod,
    >>     options={"guard_filter_fn": torch.compiler.skip_all_guards_unsafe},
    >> )
    """

COMPILER_EXPORTS = [
    "assume_constant_result",
    "reset",
    "list_backends",
    "disable",
    "set_default_backend",
    "get_default_backend",
    "set_enable_guard_collectives",
    "cudagraph_mark_step_begin",
    "is_compiling",
    "is_dynamo_compiling",
    "is_exporting",
    "keep_portable_guards_unsafe",
    "skip_guard_on_inbuilt_nn_modules_unsafe",
    "skip_guard_on_all_nn_modules_unsafe",
    "keep_tensor_guards_unsafe",
    "skip_guard_on_globals_unsafe",
    "skip_all_guards_unsafe",
]


class _OpaqueEntry:
    def _fail(self, operation):
        raise AssertionError(f"guard entry was inspected through {operation}")

    def __bool__(self):
        self._fail("bool")

    def __eq__(self, other):
        self._fail("equality")

    def __hash__(self):
        self._fail("hash")

    def __iter__(self):
        self._fail("iteration")

    def __len__(self):
        self._fail("length")

    def __repr__(self):
        self._fail("repr")

    def __str__(self):
        self._fail("str")


class _CountingIterator:
    def __init__(self, owner):
        self.owner = owner
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.owner.next_calls += 1
        if self.index == len(self.owner.entries):
            raise StopIteration
        entry = self.owner.entries[self.index]
        self.index += 1
        return entry


class _CountingIterable:
    def __init__(self, entries):
        self.entries = entries
        self.iter_calls = 0
        self.next_calls = 0

    def __iter__(self):
        self.iter_calls += 1
        if self.iter_calls != 1:
            raise AssertionError("iterable was consumed more than once")
        return _CountingIterator(self)

    def __len__(self):
        raise AssertionError("iterable length was inspected")

    def __getitem__(self, index):
        raise AssertionError("iterable was indexed")


class _IterFailure:
    def __init__(self, error):
        self.error = error
        self.iter_calls = 0

    def __iter__(self):
        self.iter_calls += 1
        raise self.error


class _NextFailure:
    def __init__(self, first_entry, error):
        self.first_entry = first_entry
        self.error = error
        self.iter_calls = 0
        self.next_calls = 0

    def __iter__(self):
        self.iter_calls += 1
        return self

    def __next__(self):
        self.next_calls += 1
        if self.next_calls == 1:
            return self.first_entry
        raise self.error


class _InvalidIterator:
    def __iter__(self):
        return []


class CompilerSkipAllGuardsUnsafeTests(unittest.TestCase):
    def assert_exact_false_list(self, result, length):
        self.assertIs(type(result), list)
        self.assertEqual(len(result), length)
        for value in result:
            self.assertIs(value, False)

    def test_lists_and_tuples_produce_fresh_exact_false_lists(self):
        function = torch.compiler.skip_all_guards_unsafe
        entries = [_OpaqueEntry(), _OpaqueEntry(), _OpaqueEntry()]
        original_entries = entries.copy()

        first = function(entries)
        second = function(entries)
        from_tuple = function(tuple(entries))
        first_empty = function([])
        second_empty = function(())

        self.assert_exact_false_list(first, 3)
        self.assert_exact_false_list(second, 3)
        self.assert_exact_false_list(from_tuple, 3)
        self.assert_exact_false_list(first_empty, 0)
        self.assert_exact_false_list(second_empty, 0)
        self.assertIsNot(first, second)
        self.assertIsNot(first, entries)
        self.assertIsNot(first_empty, second_empty)
        self.assertEqual(len(entries), len(original_entries))
        for actual, original in zip(entries, original_entries):
            self.assertIs(actual, original)

    def test_custom_iterable_is_consumed_once_without_length_or_indexing(self):
        entries = [_OpaqueEntry(), _OpaqueEntry(), _OpaqueEntry(), _OpaqueEntry()]
        iterable = _CountingIterable(entries)

        result = torch.compiler.skip_all_guards_unsafe(iterable)

        self.assert_exact_false_list(result, len(entries))
        self.assertEqual(iterable.iter_calls, 1)
        self.assertEqual(iterable.next_calls, len(entries) + 1)

    def test_generator_is_fully_consumed_once_without_inspecting_entries(self):
        entries = [_OpaqueEntry(), _OpaqueEntry(), _OpaqueEntry()]
        events = []

        def guard_entries():
            for index, entry in enumerate(entries):
                events.append(("yield", index))
                yield entry
            events.append(("finished", len(entries)))

        generator = guard_entries()
        result = torch.compiler.skip_all_guards_unsafe(generator)

        self.assert_exact_false_list(result, len(entries))
        self.assertEqual(
            events,
            [("yield", 0), ("yield", 1), ("yield", 2), ("finished", 3)],
        )
        with self.assertRaises(StopIteration):
            next(generator)

    def test_iteration_exceptions_propagate_unchanged(self):
        function = torch.compiler.skip_all_guards_unsafe

        iter_error = RuntimeError("iter failure")
        iter_failure = _IterFailure(iter_error)
        with self.assertRaises(RuntimeError) as iter_raised:
            function(iter_failure)
        self.assertIs(iter_raised.exception, iter_error)
        self.assertEqual(iter_failure.iter_calls, 1)

        next_error = LookupError("next failure")
        next_failure = _NextFailure(_OpaqueEntry(), next_error)
        with self.assertRaises(LookupError) as next_raised:
            function(next_failure)
        self.assertIs(next_raised.exception, next_error)
        self.assertEqual(next_failure.iter_calls, 1)
        self.assertEqual(next_failure.next_calls, 2)

        with self.assertRaises(TypeError) as invalid_raised:
            function(_InvalidIterator())
        message = "iter() returned non-iterator of type 'list'"
        self.assertEqual(str(invalid_raised.exception), message)
        self.assertEqual(invalid_raised.exception.args, (message,))

    def test_signature_documentation_and_function_metadata(self):
        compiler = importlib.import_module("torch_rs.compiler")
        function = compiler.skip_all_guards_unsafe

        self.assertIs(torch.compiler, compiler)
        self.assertIs(sys.modules["torch_rs.compiler"], compiler)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "(guard_entries)")
        self.assertEqual(function.__annotations__, {})
        self.assertEqual(typing.get_type_hints(function), {})
        self.assertEqual(function.__name__, "skip_all_guards_unsafe")
        self.assertEqual(function.__qualname__, "skip_all_guards_unsafe")
        self.assertEqual(function.__module__, "torch_rs.compiler")
        self.assertIs(inspect.getmodule(function), compiler)
        self.assertEqual(
            inspect.cleandoc(function.__doc__), inspect.cleandoc(FUNCTION_DOC)
        )
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(function.__code__.co_names, ())
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

    def test_exports_copying_and_pickling_use_the_canonical_function(self):
        compiler = torch.compiler
        function = compiler.skip_all_guards_unsafe

        self.assertEqual(compiler.__all__, COMPILER_EXPORTS)
        compiler_namespace = {}
        exec("from torch_rs.compiler import *", compiler_namespace)
        self.assertEqual(
            {name for name in compiler_namespace if not name.startswith("__")},
            set(COMPILER_EXPORTS),
        )
        for name in COMPILER_EXPORTS:
            self.assertIs(compiler_namespace[name], getattr(compiler, name))

        self.assertNotIn("compiler", torch.__all__)
        self.assertNotIn("skip_all_guards_unsafe", torch.__all__)
        self.assertFalse(hasattr(torch, "skip_all_guards_unsafe"))
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("compiler", top_level_namespace)
        self.assertNotIn("skip_all_guards_unsafe", top_level_namespace)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.compiler", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_argument_errors_match_pytorch_2_13(self):
        function = torch.compiler.skip_all_guards_unsafe
        cases = (
            (
                lambda: function(),
                "skip_all_guards_unsafe() missing 1 required positional "
                "argument: 'guard_entries'",
            ),
            (
                lambda: function([], []),
                "skip_all_guards_unsafe() takes 1 positional argument but 2 "
                "were given",
            ),
            (
                lambda: function([], guard_entries=[]),
                "skip_all_guards_unsafe() got multiple values for argument "
                "'guard_entries'",
            ),
            (
                lambda: function(entries=[]),
                "skip_all_guards_unsafe() got an unexpected keyword argument "
                "'entries'",
            ),
            (lambda: function(None), "'NoneType' object is not iterable"),
            (lambda: function(1), "'int' object is not iterable"),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_calls_and_reload_preserve_compiler_and_grad_state(self):
        compiler = torch.compiler
        original_backend = compiler.get_default_backend()

        def backend(graph_module, example_inputs):
            return graph_module.forward

        try:
            compiler.set_default_backend(backend)
            expected_queries = (
                compiler.is_compiling(),
                compiler.is_dynamo_compiling(),
                compiler.is_exporting(),
            )

            for context, expected_grad_state in (
                (contextlib.nullcontext(), True),
                (torch.no_grad(), False),
            ):
                with context:
                    self.assertIs(torch.is_grad_enabled(), expected_grad_state)
                    self.assertEqual(
                        compiler.skip_all_guards_unsafe(iter((_OpaqueEntry(),))),
                        [False],
                    )
                    self.assertIs(torch.is_grad_enabled(), expected_grad_state)
                    self.assertIs(compiler.get_default_backend(), backend)
                    self.assertEqual(
                        (
                            compiler.is_compiling(),
                            compiler.is_dynamo_compiling(),
                            compiler.is_exporting(),
                        ),
                        expected_queries,
                    )

            old_function = compiler.skip_all_guards_unsafe
            old_exports = compiler.__all__
            reloaded = importlib.reload(compiler)
            new_function = reloaded.skip_all_guards_unsafe

            self.assertIs(reloaded, compiler)
            self.assertIs(torch.compiler, compiler)
            self.assertIs(sys.modules["torch_rs.compiler"], compiler)
            self.assertIsNot(new_function, old_function)
            self.assertIsNot(compiler.__all__, old_exports)
            self.assertEqual(compiler.__all__, COMPILER_EXPORTS)
            self.assertIs(compiler.get_default_backend(), backend)
            self.assertEqual(new_function((_OpaqueEntry(),)), [False])
            self.assertIs(copy.copy(old_function), old_function)
            self.assertIs(copy.deepcopy(old_function), old_function)
            with self.assertRaises(pickle.PicklingError):
                pickle.dumps(old_function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                self.assertIs(
                    pickle.loads(pickle.dumps(new_function, protocol)),
                    new_function,
                )
        finally:
            compiler.set_default_backend(original_backend)

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

modules_before_call = set(sys.modules)
guard_entries = (object() for _ in range(3))
result = torch.compiler.skip_all_guards_unsafe(guard_entries)
assert result == [False, False, False]
assert set(sys.modules) == modules_before_call
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
