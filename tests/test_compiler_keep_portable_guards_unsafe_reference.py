import contextlib
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


class _GuardEntry:
    def __init__(self, label, guard_type, is_global, events):
        self.label = label
        self._guard_type = guard_type
        self._is_global = is_global
        self.events = events

    @property
    def guard_type(self):
        self.events.append(("guard_type", self.label))
        return self._guard_type

    @property
    def is_global(self):
        self.events.append(("is_global", self.label))
        return self._is_global


class _ChangingGuardEntry:
    def __init__(self, label, guard_types, is_global, events):
        self.label = label
        self.guard_types = iter(guard_types)
        self._is_global = is_global
        self.events = events

    @property
    def guard_type(self):
        self.events.append(("guard_type", self.label))
        return next(self.guard_types)

    @property
    def is_global(self):
        self.events.append(("is_global", self.label))
        return self._is_global


class _EqualityProbe:
    def __init__(self, label, results, events):
        self.label = label
        self.results = iter(results)
        self.events = events

    def __eq__(self, other):
        self.events.append(("eq", self.label, other))
        result = next(self.results)
        if isinstance(result, BaseException):
            raise result
        return result


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


class _GuardTypeFailureEntry:
    def __init__(self, error, fail_on_call=1):
        self.error = error
        self.fail_on_call = fail_on_call
        self.guard_type_calls = 0
        self.is_global_calls = 0

    @property
    def guard_type(self):
        self.guard_type_calls += 1
        if self.guard_type_calls == self.fail_on_call:
            raise self.error
        return "OTHER"

    @property
    def is_global(self):
        self.is_global_calls += 1
        raise AssertionError("is_global should not be accessed")


class _GlobalFailureEntry:
    def __init__(self, error):
        self.error = error
        self.guard_type_calls = 0
        self.is_global_calls = 0

    @property
    def guard_type(self):
        self.guard_type_calls += 1
        return "TENSOR_MATCH"

    @property
    def is_global(self):
        self.is_global_calls += 1
        raise self.error


class _InvalidTruth:
    def __bool__(self):
        return 1


class _MissingGuardType:
    pass


class _InvalidIterator:
    def __iter__(self):
        return []


def _normalized_code_metadata(code):
    return (
        code.co_names,
        code.co_varnames,
        tuple(
            _normalized_code_metadata(constant)
            if isinstance(constant, types.CodeType)
            else constant
            for constant in code.co_consts
        ),
        code.co_freevars,
        code.co_cellvars,
    )


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CompilerKeepPortableGuardsUnsafeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "compiler.keep_portable_guards_unsafe differentials require "
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
            _GuardEntry("global", "GLOBAL_STATE", object(), events),
            _GuardEntry("shape", "SHAPE_ENV", object(), events),
            _GuardEntry("local-tensor", "TENSOR_MATCH", False, events),
            _GuardEntry("global-tensor", "TENSOR_MATCH", True, events),
            _GuardEntry("none-tensor", "TENSOR_MATCH", None, events),
            _GuardEntry("other", "OTHER", object(), events),
        ]
        original_entries = tuple(entries)
        list_result = function(entries)
        tuple_result = function(tuple(entries))
        first_empty = function([])
        second_empty = function(())

        iterable_events = []
        iterable_entries = [
            _GuardEntry("iter-global", "GLOBAL_STATE", object(), iterable_events),
            _GuardEntry("iter-tensor", "TENSOR_MATCH", False, iterable_events),
            _GuardEntry("iter-other", "OTHER", object(), iterable_events),
        ]
        iterable = _CountingIterable(iterable_entries)
        iterable_result = function(iterable)

        generator_events = []

        def generated_entries():
            for index, entry in enumerate(iterable_entries):
                generator_events.append(("yield", index))
                yield entry
            generator_events.append(("finished", len(iterable_entries)))

        generator = generated_entries()
        generator_result = function(generator)
        try:
            next(generator)
        except StopIteration:
            generator_exhausted = True
        else:
            generator_exhausted = False

        results = (
            list_result,
            tuple_result,
            first_empty,
            second_empty,
            iterable_result,
            generator_result,
        )
        return {
            "results": results,
            "result_types": tuple(type(result).__name__ for result in results),
            "exact_bools": tuple(
                tuple(value is True if value else value is False for value in result)
                for result in results
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
            "iterable_events": iterable_events,
            "iter_calls": iterable.iter_calls,
            "next_calls": iterable.next_calls,
            "generator_events": generator_events,
            "generator_exhausted": generator_exhausted,
        }

    def access_and_truth_outcome(self, function):
        changing_events = []
        changing = _ChangingGuardEntry(
            "changing", ("OTHER", "TENSOR_MATCH"), False, changing_events
        )
        changing_result = function((changing,))

        comparison_events = []
        global_result = _TruthProbe("global-comparison", False, comparison_events)
        shape_result = _TruthProbe("shape-comparison", False, comparison_events)
        tensor_result = _TruthProbe("tensor-comparison", False, comparison_events)
        guard_type = _EqualityProbe(
            "custom",
            (global_result, shape_result, tensor_result),
            comparison_events,
        )
        comparison_entry = _GuardEntry(
            "custom", guard_type, False, comparison_events
        )
        comparison_result = function((comparison_entry,))

        short_circuit_events = []
        membership_result = _TruthProbe(
            "membership", True, short_circuit_events
        )
        guard_type = _EqualityProbe(
            "short-circuit", (membership_result,), short_circuit_events
        )
        short_circuit_entry = _GuardEntry(
            "short-circuit", guard_type, object(), short_circuit_events
        )
        short_circuit_result = function((short_circuit_entry,))

        return {
            "changing_result": changing_result,
            "changing_events": changing_events,
            "comparison_result_type": type(comparison_result[0]).__name__,
            "returned_tensor_result": comparison_result[0] is tensor_result,
            "comparison_events": comparison_events,
            "short_circuit_result": short_circuit_result,
            "short_circuit_exact_true": short_circuit_result[0] is True,
            "short_circuit_events": short_circuit_events,
        }

    def exception_outcome(self, function):
        iter_error = RuntimeError("iter failure")
        iter_failure = _IterFailure(iter_error)
        iter_outcome = self.captured_error(
            lambda: function(iter_failure), iter_error
        ) + (iter_failure.iter_calls,)

        next_error = LookupError("next failure")
        next_failure = _NextFailure(
            types.SimpleNamespace(guard_type="GLOBAL_STATE"), next_error
        )
        next_outcome = self.captured_error(
            lambda: function(next_failure), next_error
        ) + (next_failure.iter_calls, next_failure.next_calls)

        first_attribute_error = KeyError("first guard_type failure")
        first_attribute_failure = _GuardTypeFailureEntry(first_attribute_error)
        untouched = _GuardTypeFailureEntry(AssertionError("should not be reached"))
        first_attribute_outcome = self.captured_error(
            lambda: function((first_attribute_failure, untouched)),
            first_attribute_error,
        ) + (
            first_attribute_failure.guard_type_calls,
            first_attribute_failure.is_global_calls,
            untouched.guard_type_calls,
        )

        second_attribute_error = IndexError("second guard_type failure")
        second_attribute_failure = _GuardTypeFailureEntry(
            second_attribute_error, fail_on_call=2
        )
        second_attribute_outcome = self.captured_error(
            lambda: function((second_attribute_failure,)), second_attribute_error
        ) + (
            second_attribute_failure.guard_type_calls,
            second_attribute_failure.is_global_calls,
        )

        comparison_error = ZeroDivisionError("comparison failure")
        comparison_value = _EqualityProbe("comparison", (comparison_error,), [])
        comparison_outcome = self.captured_error(
            lambda: function(
                (types.SimpleNamespace(guard_type=comparison_value),)
            ),
            comparison_error,
        )

        comparison_truth_error = ValueError("comparison truth failure")
        comparison_truth_events = []
        comparison_truth = _TruthProbe(
            "comparison", comparison_truth_error, comparison_truth_events
        )
        comparison_value = _EqualityProbe(
            "comparison", (comparison_truth,), comparison_truth_events
        )
        comparison_truth_outcome = self.captured_error(
            lambda: function(
                (types.SimpleNamespace(guard_type=comparison_value),)
            ),
            comparison_truth_error,
        ) + (comparison_truth_events,)

        global_error = OSError("is_global failure")
        global_failure = _GlobalFailureEntry(global_error)
        global_outcome = self.captured_error(
            lambda: function((global_failure,)), global_error
        ) + (global_failure.guard_type_calls, global_failure.is_global_calls)

        global_truth_error = ArithmeticError("is_global truth failure")
        global_truth_events = []
        global_truth = _TruthProbe(
            "is_global", global_truth_error, global_truth_events
        )
        global_truth_outcome = self.captured_error(
            lambda: function(
                (
                    types.SimpleNamespace(
                        guard_type="TENSOR_MATCH", is_global=global_truth
                    ),
                )
            ),
            global_truth_error,
        ) + (global_truth_events,)

        return (
            iter_outcome,
            next_outcome,
            first_attribute_outcome,
            second_attribute_outcome,
            comparison_outcome,
            comparison_truth_outcome,
            global_outcome,
            global_truth_outcome,
            self.captured_error(lambda: function((_MissingGuardType(),))),
            self.captured_error(
                lambda: function(
                    (
                        types.SimpleNamespace(
                            guard_type="TENSOR_MATCH", is_global=_InvalidTruth()
                        ),
                    )
                )
            ),
            self.captured_error(lambda: function(_InvalidIterator())),
        )

    def test_values_iteration_and_access_order_match_pytorch_2_13(self):
        actual = torch.compiler.keep_portable_guards_unsafe
        expected = reference_torch.compiler.keep_portable_guards_unsafe
        self.assertEqual(self.behavior_outcome(actual), self.behavior_outcome(expected))
        self.assertEqual(
            self.access_and_truth_outcome(actual),
            self.access_and_truth_outcome(expected),
        )

    def test_exceptions_and_argument_errors_match_pytorch_2_13(self):
        actual = torch.compiler.keep_portable_guards_unsafe
        expected = reference_torch.compiler.keep_portable_guards_unsafe
        self.assertEqual(
            self.exception_outcome(actual), self.exception_outcome(expected)
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
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_signature_documentation_and_metadata_match_pytorch_2_13(self):
        actual_compiler = importlib.import_module("torch_rs.compiler")
        expected_compiler = importlib.import_module("torch.compiler")
        actual = actual_compiler.keep_portable_guards_unsafe
        expected = expected_compiler.keep_portable_guards_unsafe

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
        self.assertEqual(
            _normalized_code_metadata(actual.__code__),
            _normalized_code_metadata(expected.__code__),
        )

    def test_exports_copying_and_pickling_match_pytorch_2_13(self):
        actual_compiler = torch.compiler
        expected_compiler = reference_torch.compiler
        actual = actual_compiler.keep_portable_guards_unsafe
        expected = expected_compiler.keep_portable_guards_unsafe

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
            torch.__all__.count("keep_portable_guards_unsafe"),
            reference_torch.__all__.count("keep_portable_guards_unsafe"),
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
            self.assertNotIn("keep_portable_guards_unsafe", namespace)

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
                        types.SimpleNamespace(guard_type="GLOBAL_STATE"),
                        types.SimpleNamespace(
                            guard_type="TENSOR_MATCH", is_global=False
                        ),
                    )
                    result = compiler.keep_portable_guards_unsafe(iter(entries))
                    results.append(
                        (
                            result,
                            tuple(value is True for value in result),
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
            old_function = compiler.keep_portable_guards_unsafe
            old_exports = compiler.__all__
            reloaded = importlib.reload(compiler)
            new_function = reloaded.keep_portable_guards_unsafe

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
            entry = types.SimpleNamespace(guard_type="SHAPE_ENV")
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
