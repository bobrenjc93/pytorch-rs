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
    A common function to skip guards on all globals. This is unsafe to use by
    default. But if you don't expect any changes in the globals, you can just
    keep the tensor guards.

    >> opt_mod = torch.compile(
    >>     mod,
    >>     options={"guard_filter_fn": torch.compiler.skip_guard_on_globals},
    >> )
    """

COMPILER_EXPORTS = [
    "assume_constant_result",
    "reset",
    "disable",
    "set_default_backend",
    "get_default_backend",
    "is_compiling",
    "is_dynamo_compiling",
    "is_exporting",
    "skip_guard_on_globals_unsafe",
    "skip_all_guards_unsafe",
]


class _PlainEntry:
    def __init__(self, is_global, marker):
        self.is_global = is_global
        self.marker = marker


class _TruthValue:
    def __init__(self, name, result, events=None):
        self.name = name
        self.result = result
        self.events = events
        self.calls = 0

    def __bool__(self):
        self.calls += 1
        if self.events is not None:
            self.events.append(("bool", self.name))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class _LengthTruthValue:
    def __init__(self, length):
        self.length = length
        self.calls = 0

    def __len__(self):
        self.calls += 1
        return self.length


class _ObservedEntry:
    def __init__(self, name, value, events):
        self.name = name
        self.value = value
        self.events = events
        self.lookups = 0

    @property
    def is_global(self):
        self.lookups += 1
        self.events.append(("attribute", self.name))
        if isinstance(self.value, BaseException):
            raise self.value
        return self.value


class _CountingIterator:
    def __init__(self, owner):
        self.owner = owner
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.owner.next_calls += 1
        self.owner.events.append(("next", self.index))
        if self.index == len(self.owner.entries):
            raise StopIteration
        entry = self.owner.entries[self.index]
        self.index += 1
        return entry


class _CountingIterable:
    def __init__(self, entries, events):
        self.entries = entries
        self.events = events
        self.iter_calls = 0
        self.next_calls = 0

    def __iter__(self):
        self.iter_calls += 1
        self.events.append(("iter", self.iter_calls))
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


class _MissingEntry:
    pass


class CompilerSkipGuardOnGlobalsUnsafeTests(unittest.TestCase):
    def assert_exact_bool_list(self, result, expected):
        self.assertIs(type(result), list)
        self.assertEqual(result, expected)
        for value in result:
            self.assertIs(type(value), bool)

    def test_lists_and_tuples_return_fresh_booleans_and_preserve_entries(self):
        values = (True, False, 1, 0, "global", "", [1], [], None)
        entries = [_PlainEntry(value, object()) for value in values]
        original_entries = entries.copy()
        snapshots = [entry.__dict__.copy() for entry in entries]
        expected = [False, True, False, True, False, True, False, True, True]

        first = torch.compiler.skip_guard_on_globals_unsafe(entries)
        second = torch.compiler.skip_guard_on_globals_unsafe(entries)
        from_tuple = torch.compiler.skip_guard_on_globals_unsafe(tuple(entries))
        first_empty = torch.compiler.skip_guard_on_globals_unsafe([])
        second_empty = torch.compiler.skip_guard_on_globals_unsafe(())

        for result in (first, second, from_tuple):
            self.assert_exact_bool_list(result, expected)
        self.assert_exact_bool_list(first_empty, [])
        self.assert_exact_bool_list(second_empty, [])
        self.assertIsNot(first, second)
        self.assertIsNot(first, entries)
        self.assertIsNot(first_empty, second_empty)
        for entry, original, snapshot, value in zip(
            entries, original_entries, snapshots, values
        ):
            self.assertIs(entry, original)
            self.assertEqual(entry.__dict__, snapshot)
            self.assertIs(entry.is_global, value)

    def test_iterable_is_consumed_once_with_interleaved_lookup_and_truthiness(self):
        events = []
        truth_values = [
            _TruthValue("first", True, events),
            _TruthValue("second", False, events),
            _TruthValue("third", True, events),
        ]
        entries = [
            _ObservedEntry(value.name, value, events) for value in truth_values
        ]
        iterable = _CountingIterable(entries, events)

        result = torch.compiler.skip_guard_on_globals_unsafe(iterable)

        self.assert_exact_bool_list(result, [False, True, False])
        self.assertEqual(iterable.iter_calls, 1)
        self.assertEqual(iterable.next_calls, 4)
        self.assertEqual([entry.lookups for entry in entries], [1, 1, 1])
        self.assertEqual([value.calls for value in truth_values], [1, 1, 1])
        self.assertEqual(
            events,
            [
                ("iter", 1),
                ("next", 0),
                ("attribute", "first"),
                ("bool", "first"),
                ("next", 1),
                ("attribute", "second"),
                ("bool", "second"),
                ("next", 2),
                ("attribute", "third"),
                ("bool", "third"),
                ("next", 3),
            ],
        )

    def test_truthiness_uses_bool_then_len_and_returns_exact_booleans(self):
        false_by_length = _LengthTruthValue(0)
        true_by_length = _LengthTruthValue(3)
        result = torch.compiler.skip_guard_on_globals_unsafe(
            [
                _PlainEntry(false_by_length, object()),
                _PlainEntry(true_by_length, object()),
            ]
        )

        self.assert_exact_bool_list(result, [True, False])
        self.assertEqual(false_by_length.calls, 1)
        self.assertEqual(true_by_length.calls, 1)

    def test_attribute_and_truthiness_exceptions_propagate_unchanged(self):
        function = torch.compiler.skip_guard_on_globals_unsafe

        with self.assertRaises(AttributeError) as missing_raised:
            function([_MissingEntry()])
        missing_message = "'_MissingEntry' object has no attribute 'is_global'"
        self.assertEqual(str(missing_raised.exception), missing_message)
        self.assertEqual(missing_raised.exception.args, (missing_message,))

        attribute_error = KeyError("attribute failure")
        later_events = []
        attribute_entries = [
            _ObservedEntry("failing", attribute_error, later_events),
            _ObservedEntry("unreached", True, later_events),
        ]
        with self.assertRaises(KeyError) as attribute_raised:
            function(attribute_entries)
        self.assertIs(attribute_raised.exception, attribute_error)
        self.assertEqual([entry.lookups for entry in attribute_entries], [1, 0])

        truth_error = LookupError("truth failure")
        truth_value = _TruthValue("failing", truth_error)
        truth_entry = _ObservedEntry("failing", truth_value, [])
        with self.assertRaises(LookupError) as truth_raised:
            function([truth_entry])
        self.assertIs(truth_raised.exception, truth_error)
        self.assertEqual(truth_entry.lookups, 1)
        self.assertEqual(truth_value.calls, 1)

        bad_truth = _TruthValue("bad", 1)
        with self.assertRaises(TypeError) as bad_bool_raised:
            function([_PlainEntry(bad_truth, object())])
        bad_bool_message = "__bool__ should return bool, returned int"
        self.assertEqual(str(bad_bool_raised.exception), bad_bool_message)
        self.assertEqual(bad_bool_raised.exception.args, (bad_bool_message,))

    def test_iteration_exceptions_propagate_unchanged(self):
        function = torch.compiler.skip_guard_on_globals_unsafe

        iter_error = RuntimeError("iter failure")
        iter_failure = _IterFailure(iter_error)
        with self.assertRaises(RuntimeError) as iter_raised:
            function(iter_failure)
        self.assertIs(iter_raised.exception, iter_error)
        self.assertEqual(iter_failure.iter_calls, 1)

        next_error = LookupError("next failure")
        next_failure = _NextFailure(_PlainEntry(True, object()), next_error)
        with self.assertRaises(LookupError) as next_raised:
            function(next_failure)
        self.assertIs(next_raised.exception, next_error)
        self.assertEqual(next_failure.next_calls, 2)

        with self.assertRaises(TypeError) as invalid_raised:
            function(_InvalidIterator())
        invalid_message = "iter() returned non-iterator of type 'list'"
        self.assertEqual(str(invalid_raised.exception), invalid_message)
        self.assertEqual(invalid_raised.exception.args, (invalid_message,))

    def test_signature_documentation_and_function_metadata(self):
        compiler = importlib.import_module("torch_rs.compiler")
        function = compiler.skip_guard_on_globals_unsafe

        self.assertIs(torch.compiler, compiler)
        self.assertIs(sys.modules["torch_rs.compiler"], compiler)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "(guard_entries)")
        self.assertEqual(function.__annotations__, {})
        self.assertEqual(typing.get_type_hints(function), {})
        self.assertEqual(function.__name__, "skip_guard_on_globals_unsafe")
        self.assertEqual(function.__qualname__, "skip_guard_on_globals_unsafe")
        self.assertEqual(function.__module__, "torch_rs.compiler")
        self.assertIs(inspect.getmodule(function), compiler)
        self.assertEqual(
            inspect.cleandoc(function.__doc__), inspect.cleandoc(FUNCTION_DOC)
        )
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_exports_copying_and_pickling_use_the_canonical_function(self):
        compiler = torch.compiler
        function = compiler.skip_guard_on_globals_unsafe

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
        self.assertNotIn("skip_guard_on_globals_unsafe", torch.__all__)
        self.assertFalse(hasattr(torch, "skip_guard_on_globals_unsafe"))
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("compiler", top_level_namespace)
        self.assertNotIn("skip_guard_on_globals_unsafe", top_level_namespace)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.compiler", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_argument_errors_match_pytorch_2_13(self):
        function = torch.compiler.skip_guard_on_globals_unsafe
        cases = (
            (
                lambda: function(),
                "skip_guard_on_globals_unsafe() missing 1 required positional "
                "argument: 'guard_entries'",
            ),
            (
                lambda: function([], []),
                "skip_guard_on_globals_unsafe() takes 1 positional argument but 2 "
                "were given",
            ),
            (
                lambda: function([], guard_entries=[]),
                "skip_guard_on_globals_unsafe() got multiple values for argument "
                "'guard_entries'",
            ),
            (
                lambda: function(entries=[]),
                "skip_guard_on_globals_unsafe() got an unexpected keyword argument "
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
                    result = compiler.skip_guard_on_globals_unsafe(
                        iter(
                            (
                                _PlainEntry(True, object()),
                                _PlainEntry(False, object()),
                            )
                        )
                    )
                    self.assert_exact_bool_list(result, [False, True])
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

            old_function = compiler.skip_guard_on_globals_unsafe
            old_exports = compiler.__all__
            reloaded = importlib.reload(compiler)
            new_function = reloaded.skip_guard_on_globals_unsafe

            self.assertIs(reloaded, compiler)
            self.assertIs(torch.compiler, compiler)
            self.assertIs(sys.modules["torch_rs.compiler"], compiler)
            self.assertIsNot(new_function, old_function)
            self.assertIsNot(compiler.__all__, old_exports)
            self.assertEqual(compiler.__all__, COMPILER_EXPORTS)
            self.assertIs(compiler.get_default_backend(), backend)
            self.assertEqual(new_function((_PlainEntry(True, object()),)), [False])
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

    def test_importing_and_calling_does_not_import_pytorch_or_new_modules(self):
        script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

class Entry:
    def __init__(self, is_global):
        self.is_global = is_global

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

modules_before_call = set(sys.modules)
guard_entries = (Entry(value) for value in (True, False, 1, 0))
result = torch.compiler.skip_guard_on_globals_unsafe(guard_entries)
assert result == [False, True, False, True]
assert all(type(value) is bool for value in result)
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
