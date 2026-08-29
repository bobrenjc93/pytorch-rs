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
        return self._guard_type

    @property
    def value(self):
        self.events.append(("value", self.label))
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


class _GuardTypeFailureEntry:
    def __init__(self, error):
        self.error = error
        self.value_calls = 0

    @property
    def guard_type(self):
        raise self.error

    @property
    def value(self):
        self.value_calls += 1
        raise AssertionError("value should not be accessed")


class _ValueFailureEntry:
    def __init__(self, error):
        self.error = error
        self.guard_type_calls = 0

    @property
    def guard_type(self):
        self.guard_type_calls += 1
        return "TENSOR_MATCH"

    @property
    def value(self):
        raise self.error


class _InvalidIterator:
    def __iter__(self):
        return []


class _ParameterMarker:
    pass


@contextlib.contextmanager
def _temporary_parameter_type(module):
    missing = object()
    original = getattr(module.nn, "Parameter", missing)
    module.nn.Parameter = _ParameterMarker
    try:
        yield _ParameterMarker
    finally:
        if original is missing:
            delattr(module.nn, "Parameter")
        else:
            module.nn.Parameter = original


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CompilerKeepTensorGuardsUnsafeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "compiler.keep_tensor_guards_unsafe differentials require "
                "pinned PyTorch 2.13.0"
            )
        if not hasattr(reference_torch.compiler, "keep_tensor_guards_unsafe"):
            raise AssertionError(
                "reference PyTorch lacks compiler.keep_tensor_guards_unsafe"
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

    def parameter(self, parameter_type):
        return parameter_type()

    def behavior_outcome(self, module, parameter_type):
        function = module.compiler.keep_tensor_guards_unsafe
        events = []
        entries = [
            _GuardEntry("tensor", "TENSOR_MATCH", object(), events),
            _GuardEntry(
                "parameter",
                "TENSOR_MATCH",
                self.parameter(parameter_type),
                events,
            ),
            _GuardEntry("other", "OTHER", self.parameter(parameter_type), events),
        ]
        original_entries = tuple(entries)

        default_result = function(entries)
        keep_parameter_result = function(tuple(entries), keep_parameters=True)
        first_empty = function([])
        second_empty = function(())

        iterable_entries = [
            types.SimpleNamespace(guard_type="TENSOR_MATCH", value=object()),
            types.SimpleNamespace(guard_type="OTHER", value=object()),
            types.SimpleNamespace(
                guard_type="TENSOR_MATCH",
                value=self.parameter(parameter_type),
            ),
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
            default_result,
            keep_parameter_result,
            first_empty,
            second_empty,
            iterable_result,
            generator_result,
        )
        return {
            "results": results,
            "result_types": tuple(type(result).__name__ for result in results),
            "exact_bools": tuple(
                tuple(value is True or value is False for value in result)
                for result in results
            ),
            "fresh": (
                default_result is not entries,
                default_result is not keep_parameter_result,
                first_empty is not second_empty,
            ),
            "source_preserved": all(
                actual is original
                for actual, original in zip(entries, original_entries)
            ),
            "events": events,
            "iter_calls": iterable.iter_calls,
            "next_calls": iterable.next_calls,
            "generator_events": generator_events,
            "generator_exhausted": generator_exhausted,
        }

    def access_outcome(self, module, parameter_type):
        function = module.compiler.keep_tensor_guards_unsafe
        events = []
        keep_parameters = _TruthProbe("keep-parameters", True, events)
        entries = (
            _GuardEntry("plain", "TENSOR_MATCH", object(), events),
            _GuardEntry(
                "parameter",
                "TENSOR_MATCH",
                self.parameter(parameter_type),
                events,
            ),
            _GuardEntry("other", "OTHER", self.parameter(parameter_type), events),
        )
        keep_result = function(entries, keep_parameters=keep_parameters)

        comparison_events = []
        comparison_result = _TruthProbe("comparison", True, comparison_events)
        guard_type = _EqualityProbe(
            "guard_type",
            comparison_result,
            comparison_events,
        )
        custom_result = function(
            (
                _GuardEntry(
                    "custom",
                    guard_type,
                    object(),
                    comparison_events,
                ),
            )
        )
        return {
            "keep_result": keep_result,
            "events": events,
            "custom_result": custom_result,
            "comparison_events": comparison_events,
        }

    def exception_outcome(self, module, parameter_type):
        function = module.compiler.keep_tensor_guards_unsafe

        iter_error = RuntimeError("iter failure")
        iter_failure = _IterFailure(iter_error)
        iter_outcome = self.captured_error(lambda: function(iter_failure), iter_error)

        first = types.SimpleNamespace(guard_type="OTHER", value=object())
        next_error = LookupError("next failure")
        next_failure = _NextFailure(first, next_error)
        next_outcome = self.captured_error(lambda: function(next_failure), next_error)

        guard_type_error = KeyError("guard_type failure")
        guard_type_failure = _GuardTypeFailureEntry(guard_type_error)
        guard_type_outcome = self.captured_error(
            lambda: function((guard_type_failure,)),
            guard_type_error,
        )

        comparison_error = ZeroDivisionError("comparison failure")
        comparison_value = _EqualityProbe("comparison", comparison_error, [])
        comparison_outcome = self.captured_error(
            lambda: function((types.SimpleNamespace(guard_type=comparison_value),)),
            comparison_error,
        )

        comparison_truth_error = ValueError("comparison truth failure")
        comparison_value = _EqualityProbe(
            "comparison",
            _TruthProbe("comparison", comparison_truth_error, []),
            [],
        )
        comparison_truth_outcome = self.captured_error(
            lambda: function((types.SimpleNamespace(guard_type=comparison_value),)),
            comparison_truth_error,
        )

        value_error = OSError("value failure")
        value_failure = _ValueFailureEntry(value_error)
        value_outcome = self.captured_error(
            lambda: function((value_failure,)),
            value_error,
        )

        keep_parameters_error = ArithmeticError("keep_parameters failure")
        keep_parameters_outcome = self.captured_error(
            lambda: function(
                (
                    types.SimpleNamespace(
                        guard_type="TENSOR_MATCH",
                        value=self.parameter(parameter_type),
                    ),
                ),
                keep_parameters=_TruthProbe(
                    "keep-parameters",
                    keep_parameters_error,
                    [],
                ),
            ),
            keep_parameters_error,
        )

        missing = object()
        original_parameter = getattr(module.nn, "Parameter", missing)
        try:
            module.nn.Parameter = object()
            invalid_parameter_outcome = self.captured_error(
                lambda: function(
                    (
                        types.SimpleNamespace(
                            guard_type="TENSOR_MATCH",
                            value=object(),
                        ),
                    )
                )
            )
        finally:
            if original_parameter is missing:
                delattr(module.nn, "Parameter")
            else:
                module.nn.Parameter = original_parameter

        invalid_iterator_outcome = self.captured_error(
            lambda: function(_InvalidIterator())
        )

        return (
            iter_outcome,
            iter_failure.iter_calls,
            next_outcome,
            next_failure.iter_calls,
            next_failure.next_calls,
            guard_type_outcome,
            guard_type_failure.value_calls,
            comparison_outcome,
            comparison_truth_outcome,
            value_outcome,
            value_failure.guard_type_calls,
            keep_parameters_outcome,
            invalid_parameter_outcome,
            invalid_iterator_outcome,
        )

    def test_values_iteration_and_access_order_match_pytorch_2_13(self):
        with (
            _temporary_parameter_type(torch) as actual_parameter,
            _temporary_parameter_type(reference_torch) as expected_parameter,
        ):
            self.assertEqual(
                self.behavior_outcome(torch, actual_parameter),
                self.behavior_outcome(reference_torch, expected_parameter),
            )
            self.assertEqual(
                self.access_outcome(torch, actual_parameter),
                self.access_outcome(reference_torch, expected_parameter),
            )

    def test_exceptions_and_argument_errors_match_pytorch_2_13(self):
        with (
            _temporary_parameter_type(torch) as actual_parameter,
            _temporary_parameter_type(reference_torch) as expected_parameter,
        ):
            self.assertEqual(
                self.exception_outcome(torch, actual_parameter),
                self.exception_outcome(reference_torch, expected_parameter),
            )

        actual = torch.compiler.keep_tensor_guards_unsafe
        expected = reference_torch.compiler.keep_tensor_guards_unsafe
        cases = (
            (lambda: actual(), lambda: expected()),
            (lambda: actual([], False, None), lambda: expected([], False, None)),
            (
                lambda: actual([], guard_entries=[]),
                lambda: expected([], guard_entries=[]),
            ),
            (
                lambda: actual([], False, keep_parameters=True),
                lambda: expected([], False, keep_parameters=True),
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
        actual = actual_compiler.keep_tensor_guards_unsafe
        expected = expected_compiler.keep_tensor_guards_unsafe

        self.assertIs(torch.compiler, actual_compiler)
        self.assertIs(reference_torch.compiler, expected_compiler)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual)),
            str(inspect.signature(expected)),
        )
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(typing.get_type_hints(actual), typing.get_type_hints(expected))
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"),
            expected.__module__,
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
            torch.__all__.count("compiler"),
            reference_torch.__all__.count("compiler"),
        )
        self.assertEqual(
            torch.__all__.count("keep_tensor_guards_unsafe"),
            reference_torch.__all__.count("keep_tensor_guards_unsafe"),
        )

        for compiler in (actual_compiler, expected_compiler):
            namespace = {}
            exec(f"from {compiler.__name__} import *", namespace)
            for name in SUPPORTED_COMPILER_EXPORTS:
                self.assertIs(namespace[name], getattr(compiler, name))

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

    def state_outcome(self, module, parameter_type):
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
                    result = compiler.keep_tensor_guards_unsafe(
                        (
                            types.SimpleNamespace(
                                guard_type="TENSOR_MATCH",
                                value=self.parameter(parameter_type),
                            ),
                        ),
                        keep_parameters=True,
                    )
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
        with (
            _temporary_parameter_type(torch) as actual_parameter,
            _temporary_parameter_type(reference_torch) as expected_parameter,
        ):
            self.assertEqual(
                self.state_outcome(torch, actual_parameter),
                self.state_outcome(reference_torch, expected_parameter),
            )

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
            entry = types.SimpleNamespace(guard_type="TENSOR_MATCH", value=object())
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
