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
    A common function to keep tensor guards on all tensors. This is unsafe to
    use by default. But if you don't expect any changes in the model code, you
    can just keep the tensor guards.


    >> opt_mod = torch.compile(
    >>     mod,
    >>     options={"guard_filter_fn": torch.compiler.keep_tensor_guards},
    >> )
    """

COMPILER_EXPORTS = [
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
]


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
def _temporary_parameter_type():
    missing = object()
    original = getattr(torch.nn, "Parameter", missing)
    torch.nn.Parameter = _ParameterMarker
    try:
        yield _ParameterMarker
    finally:
        if original is missing:
            delattr(torch.nn, "Parameter")
        else:
            torch.nn.Parameter = original


def _walk_code_objects(code):
    yield code
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            yield from _walk_code_objects(constant)


class CompilerKeepTensorGuardsUnsafeTests(unittest.TestCase):
    def assert_exact_bool_list(self, result, expected):
        self.assertIs(type(result), list)
        self.assertEqual(len(result), len(expected))
        for actual, expected_value in zip(result, expected):
            self.assertIs(actual, expected_value)

    def test_parameter_api_remains_unsupported(self):
        self.assertFalse(hasattr(torch.nn, "Parameter"))
        with self.assertRaises(AttributeError):
            torch.nn.Parameter

        result = torch.compiler.keep_tensor_guards_unsafe(
            (
                types.SimpleNamespace(guard_type="TENSOR_MATCH", value=object()),
                types.SimpleNamespace(guard_type="OTHER", value=object()),
            ),
            keep_parameters=True,
        )
        self.assert_exact_bool_list(result, [True, False])
        self.assertFalse(hasattr(torch.nn, "Parameter"))

    def test_tensor_predicate_and_parameter_option_return_exact_booleans(self):
        function = torch.compiler.keep_tensor_guards_unsafe
        with _temporary_parameter_type() as Parameter:
            parameter = Parameter()
            plain_value = object()
            untouched_value = object()
            events = []
            entries = [
                _GuardEntry("tensor", "TENSOR_MATCH", plain_value, events),
                _GuardEntry("parameter", "TENSOR_MATCH", parameter, events),
                _GuardEntry("other", "OTHER", untouched_value, events),
            ]
            original_entries = tuple(entries)

            default_result = function(entries)
            keep_parameter_result = function(tuple(entries), keep_parameters=True)

        self.assert_exact_bool_list(default_result, [True, False, False])
        self.assert_exact_bool_list(keep_parameter_result, [True, True, False])
        self.assertIsNot(default_result, entries)
        self.assertIsNot(default_result, keep_parameter_result)
        self.assertTrue(
            all(
                actual is original
                for actual, original in zip(entries, original_entries)
            )
        )
        self.assertEqual(
            events,
            [
                ("guard_type", "tensor"),
                ("value", "tensor"),
                ("guard_type", "parameter"),
                ("value", "parameter"),
                ("guard_type", "other"),
                ("guard_type", "tensor"),
                ("value", "tensor"),
                ("guard_type", "parameter"),
                ("value", "parameter"),
                ("guard_type", "other"),
            ],
        )

    def test_left_to_right_access_and_keep_parameter_truthiness(self):
        with _temporary_parameter_type() as Parameter:
            parameter = Parameter()
            events = []
            keep_parameters = _TruthProbe("keep-parameters", True, events)
            entries = (
                _GuardEntry("plain", "TENSOR_MATCH", object(), events),
                _GuardEntry("parameter", "TENSOR_MATCH", parameter, events),
                _GuardEntry("other", "OTHER", parameter, events),
            )

            result = torch.compiler.keep_tensor_guards_unsafe(
                entries,
                keep_parameters=keep_parameters,
            )

        self.assert_exact_bool_list(result, [True, True, False])
        self.assertEqual(
            events,
            [
                ("guard_type", "plain"),
                ("value", "plain"),
                ("guard_type", "parameter"),
                ("value", "parameter"),
                ("bool", "keep-parameters"),
                ("guard_type", "other"),
            ],
        )

        with _temporary_parameter_type():
            events = []
            comparison_result = _TruthProbe("comparison", True, events)
            guard_type = _EqualityProbe("guard_type", comparison_result, events)
            entry = _GuardEntry("custom", guard_type, object(), events)

            result = torch.compiler.keep_tensor_guards_unsafe((entry,))

        self.assert_exact_bool_list(result, [True])
        self.assertEqual(
            events,
            [
                ("guard_type", "custom"),
                ("eq", "guard_type", "TENSOR_MATCH"),
                ("bool", "comparison"),
                ("value", "custom"),
            ],
        )

    def test_arbitrary_iterables_are_consumed_once_in_order(self):
        with _temporary_parameter_type() as Parameter:
            entries = [
                types.SimpleNamespace(guard_type="TENSOR_MATCH", value=object()),
                types.SimpleNamespace(guard_type="OTHER", value=object()),
                types.SimpleNamespace(
                    guard_type="TENSOR_MATCH",
                    value=Parameter(),
                ),
            ]
            iterable = _CountingIterable(entries)

            result = torch.compiler.keep_tensor_guards_unsafe(iterable)

        self.assert_exact_bool_list(result, [True, False, False])
        self.assertEqual(iterable.iter_calls, 1)
        self.assertEqual(iterable.next_calls, len(entries) + 1)

        generator_events = []

        def guard_entries():
            for index, entry in enumerate(entries):
                generator_events.append(("yield", index))
                yield entry
            generator_events.append(("finished", len(entries)))

        with _temporary_parameter_type() as Parameter:
            entries[2].value = Parameter()
            generator = guard_entries()
            result = torch.compiler.keep_tensor_guards_unsafe(generator)
        self.assert_exact_bool_list(result, [True, False, False])
        self.assertEqual(
            generator_events,
            [("yield", 0), ("yield", 1), ("yield", 2), ("finished", 3)],
        )
        with self.assertRaises(StopIteration):
            next(generator)

        first_empty = torch.compiler.keep_tensor_guards_unsafe([])
        second_empty = torch.compiler.keep_tensor_guards_unsafe(())
        self.assertEqual(first_empty, [])
        self.assertEqual(second_empty, [])
        self.assertIsNot(first_empty, second_empty)

    def test_iteration_attribute_comparison_and_truth_errors_propagate(self):
        function = torch.compiler.keep_tensor_guards_unsafe

        iter_error = RuntimeError("iter failure")
        iter_failure = _IterFailure(iter_error)
        with self.assertRaises(RuntimeError) as iter_raised:
            function(iter_failure)
        self.assertIs(iter_raised.exception, iter_error)
        self.assertEqual(iter_failure.iter_calls, 1)

        first = types.SimpleNamespace(guard_type="OTHER", value=object())
        next_error = LookupError("next failure")
        next_failure = _NextFailure(first, next_error)
        with self.assertRaises(LookupError) as next_raised:
            function(next_failure)
        self.assertIs(next_raised.exception, next_error)
        self.assertEqual(next_failure.iter_calls, 1)
        self.assertEqual(next_failure.next_calls, 2)

        guard_type_error = KeyError("guard_type failure")
        guard_type_failure = _GuardTypeFailureEntry(guard_type_error)
        with self.assertRaises(KeyError) as guard_type_raised:
            function((guard_type_failure,))
        self.assertIs(guard_type_raised.exception, guard_type_error)
        self.assertEqual(guard_type_failure.value_calls, 0)

        comparison_error = ZeroDivisionError("comparison failure")
        comparison_value = _EqualityProbe("comparison", comparison_error, [])
        with self.assertRaises(ZeroDivisionError) as comparison_raised:
            function((types.SimpleNamespace(guard_type=comparison_value),))
        self.assertIs(comparison_raised.exception, comparison_error)

        comparison_truth_error = ValueError("comparison truth failure")
        comparison_value = _EqualityProbe(
            "comparison",
            _TruthProbe("comparison", comparison_truth_error, []),
            [],
        )
        with self.assertRaises(ValueError) as comparison_truth_raised:
            function((types.SimpleNamespace(guard_type=comparison_value),))
        self.assertIs(comparison_truth_raised.exception, comparison_truth_error)

        value_error = OSError("value failure")
        value_failure = _ValueFailureEntry(value_error)
        with self.assertRaises(OSError) as value_raised:
            function((value_failure,))
        self.assertIs(value_raised.exception, value_error)
        self.assertEqual(value_failure.guard_type_calls, 1)

        keep_parameters_error = ArithmeticError("keep_parameters failure")
        with _temporary_parameter_type() as Parameter:
            with self.assertRaises(ArithmeticError) as keep_parameters_raised:
                function(
                    (
                        types.SimpleNamespace(
                            guard_type="TENSOR_MATCH",
                            value=Parameter(),
                        ),
                    ),
                    keep_parameters=_TruthProbe(
                        "keep-parameters",
                        keep_parameters_error,
                        [],
                    ),
                )
        self.assertIs(keep_parameters_raised.exception, keep_parameters_error)

        missing = object()
        original_parameter = getattr(torch.nn, "Parameter", missing)
        try:
            torch.nn.Parameter = object()
            with self.assertRaises(TypeError) as parameter_raised:
                function(
                    (
                        types.SimpleNamespace(
                            guard_type="TENSOR_MATCH",
                            value=object(),
                        ),
                    )
                )
        finally:
            if original_parameter is missing:
                delattr(torch.nn, "Parameter")
            else:
                torch.nn.Parameter = original_parameter
        self.assertEqual(
            str(parameter_raised.exception),
            "isinstance() arg 2 must be a type, a tuple of types, or a union",
        )

        with self.assertRaises(TypeError) as invalid_iterator_raised:
            function(_InvalidIterator())
        invalid_iterator_message = "iter() returned non-iterator of type 'list'"
        self.assertEqual(
            str(invalid_iterator_raised.exception), invalid_iterator_message
        )

    def test_signature_documentation_and_function_metadata(self):
        compiler = importlib.import_module("torch_rs.compiler")
        function = compiler.keep_tensor_guards_unsafe

        self.assertIs(torch.compiler, compiler)
        self.assertIs(sys.modules["torch_rs.compiler"], compiler)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(function)),
            "(guard_entries, keep_parameters=False)",
        )
        self.assertEqual(function.__annotations__, {})
        self.assertEqual(typing.get_type_hints(function), {})
        self.assertEqual(function.__name__, "keep_tensor_guards_unsafe")
        self.assertEqual(function.__qualname__, "keep_tensor_guards_unsafe")
        self.assertEqual(function.__module__, "torch_rs.compiler")
        self.assertIs(inspect.getmodule(function), compiler)
        self.assertEqual(
            inspect.cleandoc(function.__doc__), inspect.cleandoc(FUNCTION_DOC)
        )
        self.assertEqual(function.__defaults__, (False,))
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))
        code_objects = tuple(_walk_code_objects(function.__code__))
        co_names = tuple(name for code in code_objects for name in code.co_names)
        self.assertEqual(
            co_names,
            (
                "guard_type",
                "value",
                "_torch",
                "nn",
                "Parameter",
                "AttributeError",
                "_MISSING_PARAMETER_TYPES",
                "isinstance",
                "append",
            ),
        )
        self.assertEqual(
            function.__code__.co_varnames[:2],
            ("guard_entries", "keep_parameters"),
        )
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

    def test_exports_copying_and_pickling_use_the_canonical_function(self):
        compiler = torch.compiler
        function = compiler.keep_tensor_guards_unsafe

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
        self.assertNotIn("keep_tensor_guards_unsafe", torch.__all__)
        self.assertFalse(hasattr(torch, "keep_tensor_guards_unsafe"))
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("compiler", top_level_namespace)
        self.assertNotIn("keep_tensor_guards_unsafe", top_level_namespace)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.compiler", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_argument_errors_match_pytorch_2_13(self):
        function = torch.compiler.keep_tensor_guards_unsafe
        cases = (
            (
                lambda: function(),
                "keep_tensor_guards_unsafe() missing 1 required positional "
                "argument: 'guard_entries'",
            ),
            (
                lambda: function([], False, None),
                "keep_tensor_guards_unsafe() takes from 1 to 2 positional "
                "arguments but 3 were given",
            ),
            (
                lambda: function([], guard_entries=[]),
                "keep_tensor_guards_unsafe() got multiple values for argument "
                "'guard_entries'",
            ),
            (
                lambda: function([], False, keep_parameters=True),
                "keep_tensor_guards_unsafe() got multiple values for argument "
                "'keep_parameters'",
            ),
            (
                lambda: function(entries=[]),
                "keep_tensor_guards_unsafe() got an unexpected keyword argument "
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
                with _temporary_parameter_type() as Parameter, context:
                    self.assertIs(torch.is_grad_enabled(), expected_grad_state)
                    entries = (
                        types.SimpleNamespace(
                            guard_type="TENSOR_MATCH",
                            value=Parameter(),
                        ),
                    )
                    self.assertEqual(
                        compiler.keep_tensor_guards_unsafe(
                            iter(entries),
                            keep_parameters=True,
                        ),
                        [True],
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

            old_function = compiler.keep_tensor_guards_unsafe
            old_exports = compiler.__all__
            reloaded = importlib.reload(compiler)
            new_function = reloaded.keep_tensor_guards_unsafe

            self.assertIs(reloaded, compiler)
            self.assertIs(torch.compiler, compiler)
            self.assertIs(sys.modules["torch_rs.compiler"], compiler)
            self.assertIsNot(new_function, old_function)
            self.assertIsNot(compiler.__all__, old_exports)
            self.assertEqual(compiler.__all__, COMPILER_EXPORTS)
            entry = types.SimpleNamespace(guard_type="TENSOR_MATCH", value=object())
            self.assertEqual(new_function((entry,)), [True])
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
import types

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

class ParameterMarker:
    pass

assert not hasattr(torch.nn, "Parameter")
torch.nn.Parameter = ParameterMarker
entries = (
    types.SimpleNamespace(guard_type="TENSOR_MATCH", value=object()),
    types.SimpleNamespace(guard_type="TENSOR_MATCH", value=ParameterMarker()),
    types.SimpleNamespace(guard_type="OTHER", value=ParameterMarker()),
)
modules_before_call = set(sys.modules)
result = torch.compiler.keep_tensor_guards_unsafe(entries)
assert result == [True, False, False]
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
