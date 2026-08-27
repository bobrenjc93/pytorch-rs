import copy
import importlib
import inspect
import pickle
import pickletools
import types
import typing
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


SUPPORTED_COMPILER_EXPORTS = {
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
}


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


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CompilerSkipGuardOnGlobalsUnsafeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "compiler.skip_guard_on_globals_unsafe differentials require "
                "pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def pickle_shape(self, function, protocol):
        shape = []
        for opcode, argument, _ in pickletools.genops(
            pickle.dumps(function, protocol=protocol)
        ):
            if opcode.name == "FRAME":
                argument = "<frame length>"
            elif isinstance(argument, str):
                argument = argument.replace("torch_rs", "torch")
            shape.append((opcode.name, argument))
        return shape

    def behavior_outcome(self, function):
        values = (True, False, 1, 0, "global", "", [1], [], None)
        entries = [_PlainEntry(value, object()) for value in values]
        originals = entries.copy()
        snapshots = [entry.__dict__.copy() for entry in entries]
        list_result = function(entries)
        second_list_result = function(entries)
        tuple_result = function(tuple(entries))
        first_empty = function([])
        second_empty = function(())

        events = []
        truth_values = [
            _TruthValue("first", True, events),
            _TruthValue("second", False, events),
            _TruthValue("third", True, events),
        ]
        observed_entries = [
            _ObservedEntry(value.name, value, events) for value in truth_values
        ]
        iterable = _CountingIterable(observed_entries, events)
        iterable_result = function(iterable)

        false_by_length = _LengthTruthValue(0)
        true_by_length = _LengthTruthValue(3)
        length_result = function(
            [
                _PlainEntry(false_by_length, object()),
                _PlainEntry(true_by_length, object()),
            ]
        )

        results = (
            list_result,
            second_list_result,
            tuple_result,
            first_empty,
            second_empty,
            iterable_result,
            length_result,
        )
        return {
            "results": results,
            "result_types": tuple(type(result).__name__ for result in results),
            "all_exact_bool": tuple(
                all(type(value) is bool for value in result) for result in results
            ),
            "fresh": (
                list_result is not second_list_result,
                list_result is not entries,
                first_empty is not second_empty,
            ),
            "preserved": all(
                entry is original and entry.__dict__ == snapshot
                for entry, original, snapshot in zip(entries, originals, snapshots)
            ),
            "iter_calls": iterable.iter_calls,
            "next_calls": iterable.next_calls,
            "lookups": tuple(entry.lookups for entry in observed_entries),
            "truth_calls": tuple(value.calls for value in truth_values),
            "length_calls": (false_by_length.calls, true_by_length.calls),
            "events": events,
        }

    def exception_outcome(self, function):
        iter_error = RuntimeError("iter failure")
        iter_failure = _IterFailure(iter_error)
        try:
            function(iter_failure)
        except BaseException as raised:
            iter_outcome = (
                type(raised).__name__,
                str(raised),
                raised.args,
                raised is iter_error,
                iter_failure.iter_calls,
            )
        else:
            self.fail("iteration exception was swallowed")

        next_error = LookupError("next failure")
        next_failure = _NextFailure(_PlainEntry(True, object()), next_error)
        try:
            function(next_failure)
        except BaseException as raised:
            next_outcome = (
                type(raised).__name__,
                str(raised),
                raised.args,
                raised is next_error,
                next_failure.iter_calls,
                next_failure.next_calls,
            )
        else:
            self.fail("next exception was swallowed")

        attribute_error = KeyError("attribute failure")
        events = []
        attribute_entries = [
            _ObservedEntry("failing", attribute_error, events),
            _ObservedEntry("unreached", True, events),
        ]
        try:
            function(attribute_entries)
        except BaseException as raised:
            attribute_outcome = (
                type(raised).__name__,
                str(raised),
                raised.args,
                raised is attribute_error,
                tuple(entry.lookups for entry in attribute_entries),
                events,
            )
        else:
            self.fail("attribute exception was swallowed")

        truth_error = LookupError("truth failure")
        truth_value = _TruthValue("failing", truth_error)
        truth_entry = _ObservedEntry("failing", truth_value, [])
        try:
            function([truth_entry])
        except BaseException as raised:
            truth_outcome = (
                type(raised).__name__,
                str(raised),
                raised.args,
                raised is truth_error,
                truth_entry.lookups,
                truth_value.calls,
            )
        else:
            self.fail("truthiness exception was swallowed")

        return iter_outcome, next_outcome, attribute_outcome, truth_outcome

    def test_iterable_attribute_and_truthiness_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.behavior_outcome(
                torch.compiler.skip_guard_on_globals_unsafe
            ),
            self.behavior_outcome(
                reference_torch.compiler.skip_guard_on_globals_unsafe
            ),
        )

    def test_iteration_attribute_and_truthiness_errors_match_pytorch_2_13(self):
        actual = torch.compiler.skip_guard_on_globals_unsafe
        expected = reference_torch.compiler.skip_guard_on_globals_unsafe

        self.assertEqual(
            self.exception_outcome(actual),
            self.exception_outcome(expected),
        )
        cases = (
            (lambda: actual(), lambda: expected()),
            (lambda: actual([], []), lambda: expected([], [])),
            (
                lambda: actual([], guard_entries=[]),
                lambda: expected([], guard_entries=[]),
            ),
            (lambda: actual(entries=[]), lambda: expected(entries=[])),
            (lambda: actual(None), lambda: expected(None)),
            (lambda: actual(1), lambda: expected(1)),
            (
                lambda: actual(_InvalidIterator()),
                lambda: expected(_InvalidIterator()),
            ),
            (lambda: actual([_MissingEntry()]), lambda: expected([_MissingEntry()])),
            (
                lambda: actual([_PlainEntry(_TruthValue("bad", 1), object())]),
                lambda: expected(
                    [_PlainEntry(_TruthValue("bad", 1), object())]
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_signature_documentation_and_metadata_match_pytorch_2_13(self):
        actual_compiler = importlib.import_module("torch_rs.compiler")
        expected_compiler = importlib.import_module("torch.compiler")
        actual = actual_compiler.skip_guard_on_globals_unsafe
        expected = expected_compiler.skip_guard_on_globals_unsafe

        self.assertIs(torch.compiler, actual_compiler)
        self.assertIs(reference_torch.compiler, expected_compiler)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual)), str(inspect.signature(expected))
        )
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(typing.get_type_hints(actual), typing.get_type_hints(expected))
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertIs(inspect.getmodule(actual), actual_compiler)
        self.assertIs(inspect.getmodule(expected), expected_compiler)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )
        self.assertEqual(actual.__code__.co_names, expected.__code__.co_names)
        self.assertEqual(actual.__code__.co_varnames, expected.__code__.co_varnames)
        self.assertEqual(actual.__code__.co_consts, expected.__code__.co_consts)
        self.assertEqual(actual.__code__.co_freevars, expected.__code__.co_freevars)
        self.assertEqual(actual.__code__.co_cellvars, expected.__code__.co_cellvars)

    def test_exports_copying_and_pickling_match_pytorch_2_13(self):
        actual_compiler = torch.compiler
        expected_compiler = reference_torch.compiler
        actual = actual_compiler.skip_guard_on_globals_unsafe
        expected = expected_compiler.skip_guard_on_globals_unsafe

        self.assertEqual(
            actual_compiler.__all__,
            [
                name
                for name in expected_compiler.__all__
                if name in SUPPORTED_COMPILER_EXPORTS
            ],
        )
        self.assertEqual(
            torch.__all__.count("compiler"),
            reference_torch.__all__.count("compiler"),
        )
        self.assertEqual(
            torch.__all__.count("skip_guard_on_globals_unsafe"),
            reference_torch.__all__.count("skip_guard_on_globals_unsafe"),
        )

        for module in (actual_compiler, expected_compiler):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            for name in SUPPORTED_COMPILER_EXPORTS:
                self.assertIs(namespace[name], getattr(module, name))

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("compiler", namespace)
            self.assertNotIn("skip_guard_on_globals_unsafe", namespace)

        for function in (actual, expected):
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(pickle.loads(pickle.dumps(expected, protocol)), expected)
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

    def state_outcome(self, module):
        compiler = module.compiler
        original_backend = compiler.get_default_backend()

        def backend(graph_module, example_inputs):
            return graph_module.forward

        try:
            compiler.set_default_backend(backend)
            before_queries = (
                compiler.is_compiling(),
                compiler.is_dynamo_compiling(),
                compiler.is_exporting(),
            )
            before_grad = module.is_grad_enabled()
            result = compiler.skip_guard_on_globals_unsafe(
                _PlainEntry(value, object()) for value in (True, False, 1, 0)
            )
            after_grad = module.is_grad_enabled()
            after_queries = (
                compiler.is_compiling(),
                compiler.is_dynamo_compiling(),
                compiler.is_exporting(),
            )
            return (
                result,
                all(type(value) is bool for value in result),
                compiler.get_default_backend() is backend,
                before_grad,
                after_grad,
                before_queries,
                after_queries,
            )
        finally:
            compiler.set_default_backend(original_backend)

    def test_calls_preserve_compiler_state_like_pytorch_2_13(self):
        self.assertEqual(
            self.state_outcome(torch),
            self.state_outcome(reference_torch),
        )

    def reload_outcome(self, module, compiler_module_name):
        compiler = importlib.import_module(compiler_module_name)
        original_backend = compiler.get_default_backend()

        def backend(graph_module, example_inputs):
            return graph_module.forward

        try:
            compiler.set_default_backend(backend)
            old_function = compiler.skip_guard_on_globals_unsafe
            old_exports = compiler.__all__
            reloaded = importlib.reload(compiler)
            new_function = reloaded.skip_guard_on_globals_unsafe

            try:
                pickle.dumps(old_function)
            except BaseException as error:
                old_pickle_error = (
                    type(error).__name__,
                    "not the same object" in str(error),
                )
            else:
                old_pickle_error = None

            new_pickle_results = tuple(
                pickle.loads(pickle.dumps(new_function, protocol)) is new_function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            )
            return (
                reloaded is compiler,
                module.compiler is compiler,
                old_function is new_function,
                old_exports is compiler.__all__,
                compiler.get_default_backend() is backend,
                new_function(
                    (_PlainEntry(True, object()), _PlainEntry(False, object()))
                ),
                copy.copy(old_function) is old_function,
                copy.deepcopy(old_function) is old_function,
                old_pickle_error,
                new_pickle_results,
            )
        finally:
            compiler.set_default_backend(original_backend)

    def test_reload_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_outcome(torch, "torch_rs.compiler"),
            self.reload_outcome(reference_torch, "torch.compiler"),
        )


if __name__ == "__main__":
    unittest.main()
