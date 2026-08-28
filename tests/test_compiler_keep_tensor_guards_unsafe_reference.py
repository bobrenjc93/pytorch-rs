import contextlib
import copy
import importlib
import inspect
import pickle
import pickletools
import types
import typing
import unittest
from unittest import mock

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
    "set_enable_guard_collectives",
    "is_compiling",
    "is_dynamo_compiling",
    "is_exporting",
    "keep_portable_guards_unsafe",
    "skip_guard_on_inbuilt_nn_modules_unsafe",
    "skip_guard_on_all_nn_modules_unsafe",
    "keep_tensor_guards_unsafe",
    "skip_guard_on_globals_unsafe",
    "skip_all_guards_unsafe",
}


class _TruthProbe:
    def __init__(self, label, result, events):
        self.label = label
        self.result = result
        self.events = events

    def __bool__(self):
        self.events.append(("bool", self.label))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class _EqualityProbe:
    def __init__(self, label, result, events):
        self.label = label
        self.result = result
        self.events = events

    def __eq__(self, other):
        self.events.append(("eq", self.label, other))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class _GuardEntry:
    def __init__(self, label, guard_type, value, events):
        self.label = label
        self._guard_type = guard_type
        self._value = value
        self.events = events

    @property
    def guard_type(self):
        self.events.append(("guard_type", self.label))
        if isinstance(self._guard_type, BaseException):
            raise self._guard_type
        return self._guard_type

    @property
    def value(self):
        self.events.append(("value", self.label))
        if isinstance(self._value, BaseException):
            raise self._value
        return self._value


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


class _Parameter:
    pass


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CompilerKeepTensorGuardsUnsafeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "compiler.keep_tensor_guards_unsafe differentials require "
                "pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(BaseException) as actual_raised:
            actual_call()
        with self.assertRaises(BaseException) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def captured_error(self, call, expected_error=None):
        try:
            call()
        except BaseException as error:
            return (
                type(error).__name__,
                str(error),
                error.args,
                error is expected_error,
            )
        self.fail("expected an exception")

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
        events = []
        entries = [
            _GuardEntry("tensor", "TENSOR_MATCH", object(), events),
            _GuardEntry("other", "OTHER", object(), events),
            _GuardEntry("tensor-two", "TENSOR_MATCH", object(), events),
        ]
        original_entries = tuple(entries)
        list_result = function(entries)
        tuple_result = function(tuple(entries), True)
        keep_probe_events = []
        keep_probe = _TruthProbe(
            "keep-parameters",
            AssertionError("must not be evaluated for non-parameters"),
            keep_probe_events,
        )
        probe_result = function(tuple(entries), keep_probe)
        first_empty = function([])
        second_empty = function((), True)

        iterable_events = []
        iterable_entries = [
            _GuardEntry("iter-tensor", "TENSOR_MATCH", object(), iterable_events),
            _GuardEntry("iter-other", "OTHER", object(), iterable_events),
        ]
        iterable = _CountingIterable(iterable_entries)
        iterable_result = function(iterable)

        return {
            "results": (
                list_result,
                tuple_result,
                probe_result,
                first_empty,
                second_empty,
                iterable_result,
            ),
            "result_types": tuple(
                type(result).__name__
                for result in (
                    list_result,
                    tuple_result,
                    probe_result,
                    first_empty,
                    second_empty,
                    iterable_result,
                )
            ),
            "fresh": (
                list_result is not entries,
                list_result is not tuple_result,
                first_empty is not second_empty,
            ),
            "source_preserved": all(
                actual is original
                for actual, original in zip(entries, original_entries)
            ),
            "events": events,
            "keep_probe_events": keep_probe_events,
            "iterable_events": iterable_events,
            "iter_calls": iterable.iter_calls,
            "next_calls": iterable.next_calls,
        }

    def comparison_outcome(self, function):
        false_events = []
        false_result = _TruthProbe("false", False, false_events)
        false_type = _EqualityProbe("other", false_result, false_events)
        false_entry = _GuardEntry(
            "other", false_type, AssertionError("unused"), false_events
        )

        true_events = []
        true_result = _TruthProbe("true", True, true_events)
        true_type = _EqualityProbe("tensor", true_result, true_events)
        true_entry = _GuardEntry("tensor", true_type, object(), true_events)

        return (
            function((false_entry,)),
            false_events,
            function((true_entry,)),
            true_events,
        )

    def exception_outcome(self, function):
        iter_error = RuntimeError("iter failure")
        iter_failure = _IterFailure(iter_error)
        iter_outcome = self.captured_error(
            lambda: function(iter_failure), iter_error
        ) + (iter_failure.iter_calls,)

        next_error = LookupError("next failure")
        next_failure = _NextFailure(
            types.SimpleNamespace(guard_type="OTHER"), next_error
        )
        next_outcome = self.captured_error(
            lambda: function(next_failure), next_error
        ) + (next_failure.iter_calls, next_failure.next_calls)

        guard_type_error = KeyError("guard_type failure")
        guard_type_entry = _GuardEntry(
            "guard", guard_type_error, object(), []
        )
        guard_type_outcome = self.captured_error(
            lambda: function((guard_type_entry,)), guard_type_error
        )

        comparison_error = ZeroDivisionError("comparison failure")
        comparison_value = _EqualityProbe("comparison", comparison_error, [])
        comparison_outcome = self.captured_error(
            lambda: function(
                (types.SimpleNamespace(guard_type=comparison_value),)
            ),
            comparison_error,
        )

        comparison_truth_error = ValueError("comparison truth failure")
        comparison_truth = _TruthProbe(
            "comparison", comparison_truth_error, []
        )
        comparison_value = _EqualityProbe("comparison", comparison_truth, [])
        comparison_truth_outcome = self.captured_error(
            lambda: function(
                (types.SimpleNamespace(guard_type=comparison_value),)
            ),
            comparison_truth_error,
        )

        value_error = OSError("value failure")
        value_entry = _GuardEntry("tensor", "TENSOR_MATCH", value_error, [])
        value_outcome = self.captured_error(
            lambda: function((value_entry,)), value_error
        )

        return (
            iter_outcome,
            next_outcome,
            guard_type_outcome,
            comparison_outcome,
            comparison_truth_outcome,
            value_outcome,
            self.captured_error(lambda: function((object(),))),
            self.captured_error(
                lambda: function(
                    (types.SimpleNamespace(guard_type="TENSOR_MATCH"),)
                )
            ),
            self.captured_error(lambda: function(_InvalidIterator())),
        )

    def parameter_outcome(self, module):
        function = module.compiler.keep_tensor_guards_unsafe
        events = []
        entries = (
            _GuardEntry("parameter", "TENSOR_MATCH", _Parameter(), events),
            _GuardEntry("object", "TENSOR_MATCH", object(), events),
            _GuardEntry("other", "OTHER", object(), events),
        )
        true_probe = _TruthProbe("true", True, events)
        false_probe = _TruthProbe("false", False, events)
        with mock.patch.object(module.nn, "Parameter", _Parameter, create=True):
            results = (
                function(entries),
                function(entries, True),
                function(entries, keep_parameters=true_probe),
                function(entries, keep_parameters=false_probe),
            )
        return results, events

    def test_behavior_and_access_order_match_pytorch_2_13(self):
        actual = torch.compiler.keep_tensor_guards_unsafe
        expected = reference_torch.compiler.keep_tensor_guards_unsafe
        self.assertEqual(self.behavior_outcome(actual), self.behavior_outcome(expected))
        self.assertEqual(
            self.comparison_outcome(actual), self.comparison_outcome(expected)
        )

    def test_parameter_branch_matches_when_parameter_type_is_available(self):
        original_reference_parameter = reference_torch.nn.Parameter
        self.assertFalse(hasattr(torch.nn, "Parameter"))
        self.assertEqual(
            self.parameter_outcome(torch), self.parameter_outcome(reference_torch)
        )
        self.assertFalse(hasattr(torch.nn, "Parameter"))
        self.assertIs(reference_torch.nn.Parameter, original_reference_parameter)

    def test_exceptions_and_argument_errors_match_pytorch_2_13(self):
        actual = torch.compiler.keep_tensor_guards_unsafe
        expected = reference_torch.compiler.keep_tensor_guards_unsafe
        self.assertEqual(
            self.exception_outcome(actual), self.exception_outcome(expected)
        )

        cases = (
            (lambda: actual(), lambda: expected()),
            (lambda: actual([], False, True), lambda: expected([], False, True)),
            (
                lambda: actual([], guard_entries=[]),
                lambda: expected([], guard_entries=[]),
            ),
            (
                lambda: actual([], False, keep_parameters=True),
                lambda: expected([], False, keep_parameters=True),
            ),
            (
                lambda: actual([], keep_parameter=True),
                lambda: expected([], keep_parameter=True),
            ),
            (lambda: actual(None), lambda: expected(None)),
            (lambda: actual(1), lambda: expected(1)),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_signature_documentation_and_metadata_match_pytorch_2_13(self):
        actual_compiler = importlib.import_module("torch_rs.compiler")
        expected_compiler = importlib.import_module("torch.compiler")
        actual = actual_compiler.keep_tensor_guards_unsafe
        expected = expected_compiler.keep_tensor_guards_unsafe

        self.assertIs(torch.compiler, actual_compiler)
        self.assertIs(reference_torch.compiler, expected_compiler)
        self.assertIs(actual_compiler.torch, torch)
        self.assertIs(expected_compiler.torch, reference_torch)
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
        self.assertEqual(actual.__code__.co_argcount, expected.__code__.co_argcount)
        self.assertEqual(
            actual.__code__.co_posonlyargcount,
            expected.__code__.co_posonlyargcount,
        )
        self.assertEqual(
            actual.__code__.co_kwonlyargcount,
            expected.__code__.co_kwonlyargcount,
        )
        self.assertEqual(actual.__code__.co_varnames, expected.__code__.co_varnames)
        self.assertEqual(actual.__code__.co_freevars, expected.__code__.co_freevars)
        self.assertEqual(actual.__code__.co_cellvars, expected.__code__.co_cellvars)

    def test_exports_copying_and_pickling_match_pytorch_2_13(self):
        actual_compiler = torch.compiler
        expected_compiler = reference_torch.compiler
        actual = actual_compiler.keep_tensor_guards_unsafe
        expected = expected_compiler.keep_tensor_guards_unsafe

        self.assertEqual(
            actual_compiler.__all__,
            [
                name
                for name in expected_compiler.__all__
                if name in SUPPORTED_COMPILER_EXPORTS
            ],
        )
        self.assertEqual(
            torch.__all__.count("keep_tensor_guards_unsafe"),
            reference_torch.__all__.count("keep_tensor_guards_unsafe"),
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
            self.assertNotIn("keep_tensor_guards_unsafe", namespace)

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
            results = []
            for context in (contextlib.nullcontext(), module.no_grad()):
                with context:
                    before_grad = module.is_grad_enabled()
                    before_queries = (
                        compiler.is_compiling(),
                        compiler.is_dynamo_compiling(),
                        compiler.is_exporting(),
                    )
                    entries = (
                        types.SimpleNamespace(
                            guard_type="TENSOR_MATCH", value=object()
                        ),
                        types.SimpleNamespace(guard_type="OTHER"),
                    )
                    result = compiler.keep_tensor_guards_unsafe(iter(entries))
                    results.append(
                        (
                            result,
                            before_grad,
                            module.is_grad_enabled(),
                            before_queries,
                            (
                                compiler.is_compiling(),
                                compiler.is_dynamo_compiling(),
                                compiler.is_exporting(),
                            ),
                            compiler.get_default_backend() is backend,
                        )
                    )
            return results
        finally:
            compiler.set_default_backend(original_backend)

    def test_calls_preserve_compiler_and_grad_state_like_pytorch_2_13(self):
        self.assertEqual(self.state_outcome(torch), self.state_outcome(reference_torch))

    def reload_outcome(self, module, compiler_module_name):
        compiler = importlib.import_module(compiler_module_name)
        original_backend = compiler.get_default_backend()

        def backend(graph_module, example_inputs):
            return graph_module.forward

        try:
            compiler.set_default_backend(backend)
            old_function = compiler.keep_tensor_guards_unsafe
            old_exports = compiler.__all__
            reloaded = importlib.reload(compiler)
            new_function = reloaded.keep_tensor_guards_unsafe

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
            entry = types.SimpleNamespace(
                guard_type="TENSOR_MATCH", value=object()
            )
            return (
                reloaded is compiler,
                module.compiler is compiler,
                old_function is new_function,
                old_exports is compiler.__all__,
                compiler.get_default_backend() is backend,
                new_function((entry,)),
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
