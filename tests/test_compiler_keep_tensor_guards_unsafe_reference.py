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
    def __init__(self, error):
        self.error = error
        self.guard_type_calls = 0
        self.value_calls = 0

    @property
    def guard_type(self):
        self.guard_type_calls += 1
        raise self.error

    @property
    def value(self):
        self.value_calls += 1
        raise AssertionError("value should not be accessed")


class _ValueFailureEntry:
    def __init__(self, error):
        self.error = error
        self.guard_type_calls = 0
        self.value_calls = 0

    @property
    def guard_type(self):
        self.guard_type_calls += 1
        return "TENSOR_MATCH"

    @property
    def value(self):
        self.value_calls += 1
        raise self.error


class _InvalidTruth:
    def __bool__(self):
        return 1


class _MissingGuardType:
    pass


class _InvalidIterator:
    def __iter__(self):
        return []


class _ParameterNamespace:
    def __init__(self, parameter_type, events):
        self.parameter_type = parameter_type
        self.events = events

    @property
    def Parameter(self):
        self.events.append(("Parameter",))
        return self.parameter_type


class _TorchNamespace:
    def __init__(self, parameter_type, events):
        self.parameter_namespace = _ParameterNamespace(parameter_type, events)
        self.events = events

    @property
    def nn(self):
        self.events.append(("nn",))
        return self.parameter_namespace


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

    def behavior_outcome(self, module):
        function = module.compiler.keep_tensor_guards_unsafe
        parameter = module.nn.Parameter()
        events = []
        entries = [
            _GuardEntry("parameter", "TENSOR_MATCH", parameter, events),
            _GuardEntry("plain", "TENSOR_MATCH", object(), events),
            _GuardEntry("none", "TENSOR_MATCH", None, events),
            _GuardEntry("shape", "SHAPE_ENV", parameter, events),
            _GuardEntry("global", "GLOBAL_STATE", parameter, events),
            _GuardEntry("other", "OTHER", parameter, events),
        ]
        original_entries = tuple(entries)
        list_result = function(entries)
        tuple_result = function(tuple(entries), True)
        first_empty = function([])
        second_empty = function(())

        iterable_events = []
        iterable_entries = [
            _GuardEntry(
                "iter-parameter", "TENSOR_MATCH", module.nn.Parameter(), iterable_events
            ),
            _GuardEntry("iter-plain", "TENSOR_MATCH", object(), iterable_events),
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
        generator_result = function(generator, True)
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

    def access_and_truth_outcome(self, module):
        function = module.compiler.keep_tensor_guards_unsafe
        events = []
        keep_parameters = _TruthProbe("keep_parameters", True, events)
        entries = (
            _GuardEntry("parameter", "TENSOR_MATCH", module.nn.Parameter(), events),
            _GuardEntry("plain", "TENSOR_MATCH", object(), events),
            _GuardEntry("other", "OTHER", module.nn.Parameter(), events),
        )
        keep_result = function(entries, keep_parameters)

        comparison_events = []
        tensor_match = _TruthProbe("tensor-match", True, comparison_events)
        guard_type = _EqualityProbe("custom", (tensor_match,), comparison_events)
        comparison_entry = _GuardEntry(
            "custom", guard_type, object(), comparison_events
        )
        comparison_result = function((comparison_entry,))

        original_torch = module.compiler.torch
        lookup_events = []
        try:
            module.compiler.torch = _TorchNamespace(module.nn.Parameter, lookup_events)
            lookup_entry = _GuardEntry(
                "parameter", "TENSOR_MATCH", module.nn.Parameter(), lookup_events
            )
            lookup_result = function((lookup_entry,), True)
        finally:
            module.compiler.torch = original_torch

        return {
            "keep_result": keep_result,
            "keep_events": events,
            "comparison_result": comparison_result,
            "comparison_events": comparison_events,
            "lookup_result": lookup_result,
            "lookup_events": lookup_events,
        }

    def exception_outcome(self, module):
        function = module.compiler.keep_tensor_guards_unsafe

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
        guard_type_failure = _GuardTypeFailureEntry(guard_type_error)
        untouched = _GuardTypeFailureEntry(AssertionError("should not be reached"))
        guard_type_outcome = self.captured_error(
            lambda: function((guard_type_failure, untouched)), guard_type_error
        ) + (
            guard_type_failure.guard_type_calls,
            guard_type_failure.value_calls,
            untouched.guard_type_calls,
        )

        comparison_error = ZeroDivisionError("comparison failure")
        comparison_value = _EqualityProbe("comparison", (comparison_error,), [])
        comparison_outcome = self.captured_error(
            lambda: function((types.SimpleNamespace(guard_type=comparison_value),)),
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
            lambda: function((types.SimpleNamespace(guard_type=comparison_value),)),
            comparison_truth_error,
        ) + (comparison_truth_events,)

        value_error = OSError("value failure")
        value_failure = _ValueFailureEntry(value_error)
        value_outcome = self.captured_error(
            lambda: function((value_failure,)), value_error
        ) + (value_failure.guard_type_calls, value_failure.value_calls)

        keep_error = ArithmeticError("keep_parameters failure")
        keep_truth_events = []
        keep_truth = _TruthProbe(
            "keep_parameters", keep_error, keep_truth_events
        )
        keep_outcome = self.captured_error(
            lambda: function(
                (
                    types.SimpleNamespace(
                        guard_type="TENSOR_MATCH", value=module.nn.Parameter()
                    ),
                ),
                keep_truth,
            ),
            keep_error,
        ) + (keep_truth_events,)

        original_parameter = module.nn.Parameter
        try:
            module.nn.Parameter = 42
            invalid_parameter_outcome = self.captured_error(
                lambda: function(
                    (types.SimpleNamespace(guard_type="TENSOR_MATCH", value=object()),)
                )
            )
        finally:
            module.nn.Parameter = original_parameter

        return (
            iter_outcome,
            next_outcome,
            guard_type_outcome,
            comparison_outcome,
            comparison_truth_outcome,
            value_outcome,
            keep_outcome,
            invalid_parameter_outcome,
            self.captured_error(lambda: function((_MissingGuardType(),))),
            self.captured_error(
                lambda: function(
                    (
                        types.SimpleNamespace(
                            guard_type="TENSOR_MATCH", value=module.nn.Parameter()
                        ),
                    ),
                    _InvalidTruth(),
                )
            ),
            self.captured_error(lambda: function(_InvalidIterator())),
        )

    def test_values_iteration_and_access_order_match_pytorch_2_13(self):
        self.assertEqual(
            self.behavior_outcome(torch), self.behavior_outcome(reference_torch)
        )
        self.assertEqual(
            self.access_and_truth_outcome(torch),
            self.access_and_truth_outcome(reference_torch),
        )

    def test_exceptions_and_argument_errors_match_pytorch_2_13(self):
        actual = torch.compiler.keep_tensor_guards_unsafe
        expected = reference_torch.compiler.keep_tensor_guards_unsafe
        self.assertEqual(self.exception_outcome(torch), self.exception_outcome(reference_torch))

        cases = (
            (lambda: actual(), lambda: expected()),
            (lambda: actual([], False, None), lambda: expected([], False, None)),
            (
                lambda: actual([], [], keep_parameters=False),
                lambda: expected([], [], keep_parameters=False),
            ),
            (
                lambda: actual([], keep_params=False),
                lambda: expected([], keep_params=False),
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
                            guard_type="TENSOR_MATCH", value=module.nn.Parameter()
                        ),
                        types.SimpleNamespace(
                            guard_type="TENSOR_MATCH", value=object()
                        ),
                    )
                    result = compiler.keep_tensor_guards_unsafe(iter(entries), True)
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

    def test_compile_export_and_other_compiler_apis_remain_unsupported(self):
        self.assertTrue(callable(reference_torch.compile))
        self.assertTrue(hasattr(reference_torch, "export"))
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch, "export"))

        unsupported = set(reference_torch.compiler.__all__) - SUPPORTED_COMPILER_EXPORTS
        self.assertTrue(unsupported)
        for name in unsupported:
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.compiler, name))


if __name__ == "__main__":
    unittest.main()
