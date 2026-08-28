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
from unittest import mock

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


class CompilerKeepTensorGuardsUnsafeTests(unittest.TestCase):
    def assert_exact_bool_list(self, result, expected):
        self.assertIs(type(result), list)
        self.assertEqual(len(result), len(expected))
        for actual, expected_value in zip(result, expected):
            self.assertIs(actual, expected_value)

    def test_current_no_parameter_surface_keeps_tensor_matches(self):
        self.assertFalse(hasattr(torch.nn, "Parameter"))
        events = []
        entries = [
            _GuardEntry("tensor-object", "TENSOR_MATCH", object(), events),
            _GuardEntry("other", "OTHER", AssertionError("unused"), events),
            _GuardEntry("tensor", "TENSOR_MATCH", torch.tensor(1.0), events),
        ]
        original_entries = tuple(entries)

        result = torch.compiler.keep_tensor_guards_unsafe(entries)

        self.assert_exact_bool_list(result, [True, False, True])
        self.assertIsNot(result, entries)
        self.assertTrue(
            all(
                actual is original
                for actual, original in zip(entries, original_entries)
            )
        )
        self.assertEqual(
            events,
            [
                ("guard_type", "tensor-object"),
                ("value", "tensor-object"),
                ("guard_type", "other"),
                ("guard_type", "tensor"),
                ("value", "tensor"),
            ],
        )

        keep_parameters = _TruthProbe(
            "keep-parameters", AssertionError("must not be evaluated"), []
        )
        self.assert_exact_bool_list(
            torch.compiler.keep_tensor_guards_unsafe(
                (types.SimpleNamespace(guard_type="TENSOR_MATCH", value=object()),),
                keep_parameters,
            ),
            [True],
        )

    def test_parameter_branch_and_keep_parameters_truth(self):
        events = []
        keep_parameters = _TruthProbe("keep-parameters", True, events)
        entries = (
            _GuardEntry("parameter", "TENSOR_MATCH", _Parameter(), events),
            _GuardEntry("object", "TENSOR_MATCH", object(), events),
            _GuardEntry("other", "OTHER", AssertionError("unused"), events),
        )

        with mock.patch.object(torch.nn, "Parameter", _Parameter, create=True):
            self.assert_exact_bool_list(
                torch.compiler.keep_tensor_guards_unsafe(entries),
                [False, True, False],
            )
            self.assert_exact_bool_list(
                torch.compiler.keep_tensor_guards_unsafe(entries, keep_parameters),
                [True, True, False],
            )

        self.assertFalse(hasattr(torch.nn, "Parameter"))
        self.assertEqual(events.count(("bool", "keep-parameters")), 1)

    def test_comparison_short_circuits_value_access(self):
        events = []
        false_comparison = _TruthProbe("comparison", False, events)
        false_guard_type = _EqualityProbe("other", false_comparison, events)
        false_entry = _GuardEntry(
            "other", false_guard_type, AssertionError("unused"), events
        )

        self.assert_exact_bool_list(
            torch.compiler.keep_tensor_guards_unsafe((false_entry,)), [False]
        )
        self.assertEqual(
            events,
            [
                ("guard_type", "other"),
                ("eq", "other", "TENSOR_MATCH"),
                ("bool", "comparison"),
            ],
        )

        events = []
        true_comparison = _TruthProbe("comparison", True, events)
        true_guard_type = _EqualityProbe("tensor", true_comparison, events)
        true_entry = _GuardEntry("tensor", true_guard_type, object(), events)
        self.assert_exact_bool_list(
            torch.compiler.keep_tensor_guards_unsafe((true_entry,)), [True]
        )
        self.assertEqual(
            events,
            [
                ("guard_type", "tensor"),
                ("eq", "tensor", "TENSOR_MATCH"),
                ("bool", "comparison"),
                ("value", "tensor"),
            ],
        )

    def test_arbitrary_iterables_are_consumed_once_and_results_are_fresh(self):
        events = []
        entries = [
            _GuardEntry("tensor", "TENSOR_MATCH", object(), events),
            _GuardEntry("other", "OTHER", object(), events),
        ]
        iterable = _CountingIterable(entries)

        result = torch.compiler.keep_tensor_guards_unsafe(iterable)

        self.assert_exact_bool_list(result, [True, False])
        self.assertEqual(iterable.iter_calls, 1)
        self.assertEqual(iterable.next_calls, 3)

        yielded = []

        def guard_entries():
            for index, entry in enumerate(entries):
                yielded.append(index)
                yield entry

        generator = guard_entries()
        self.assert_exact_bool_list(
            torch.compiler.keep_tensor_guards_unsafe(generator), [True, False]
        )
        self.assertEqual(yielded, [0, 1])
        with self.assertRaises(StopIteration):
            next(generator)

        first_empty = torch.compiler.keep_tensor_guards_unsafe([])
        second_empty = torch.compiler.keep_tensor_guards_unsafe(())
        self.assertEqual(first_empty, [])
        self.assertEqual(second_empty, [])
        self.assertIsNot(first_empty, second_empty)

    def test_iteration_access_comparison_and_truth_errors_propagate(self):
        function = torch.compiler.keep_tensor_guards_unsafe

        iter_error = RuntimeError("iter failure")
        iter_failure = _IterFailure(iter_error)
        with self.assertRaises(RuntimeError) as raised:
            function(iter_failure)
        self.assertIs(raised.exception, iter_error)
        self.assertEqual(iter_failure.iter_calls, 1)

        next_error = LookupError("next failure")
        next_failure = _NextFailure(
            types.SimpleNamespace(guard_type="OTHER"), next_error
        )
        with self.assertRaises(LookupError) as raised:
            function(next_failure)
        self.assertIs(raised.exception, next_error)
        self.assertEqual((next_failure.iter_calls, next_failure.next_calls), (1, 2))

        guard_type_error = KeyError("guard_type failure")
        with self.assertRaises(KeyError) as raised:
            function((_GuardEntry("guard", guard_type_error, object(), []),))
        self.assertIs(raised.exception, guard_type_error)

        comparison_error = ZeroDivisionError("comparison failure")
        comparison_value = _EqualityProbe("comparison", comparison_error, [])
        with self.assertRaises(ZeroDivisionError) as raised:
            function((types.SimpleNamespace(guard_type=comparison_value),))
        self.assertIs(raised.exception, comparison_error)

        truth_error = ValueError("comparison truth failure")
        truth_value = _TruthProbe("comparison", truth_error, [])
        comparison_value = _EqualityProbe("comparison", truth_value, [])
        with self.assertRaises(ValueError) as raised:
            function((types.SimpleNamespace(guard_type=comparison_value),))
        self.assertIs(raised.exception, truth_error)

        value_error = OSError("value failure")
        with self.assertRaises(OSError) as raised:
            function((_GuardEntry("tensor", "TENSOR_MATCH", value_error, []),))
        self.assertIs(raised.exception, value_error)

        keep_error = ArithmeticError("keep_parameters failure")
        keep_parameters = _TruthProbe("keep-parameters", keep_error, [])
        with mock.patch.object(torch.nn, "Parameter", _Parameter, create=True):
            with self.assertRaises(ArithmeticError) as raised:
                function(
                    (
                        types.SimpleNamespace(
                            guard_type="TENSOR_MATCH", value=_Parameter()
                        ),
                    ),
                    keep_parameters,
                )
        self.assertIs(raised.exception, keep_error)

        with self.assertRaises(AttributeError):
            function((object(),))
        with self.assertRaises(AttributeError):
            function((types.SimpleNamespace(guard_type="TENSOR_MATCH"),))
        with self.assertRaises(TypeError) as raised:
            function(_InvalidIterator())
        self.assertEqual(
            str(raised.exception), "iter() returned non-iterator of type 'list'"
        )

    def test_signature_documentation_and_function_metadata(self):
        compiler = importlib.import_module("torch_rs.compiler")
        function = compiler.keep_tensor_guards_unsafe

        self.assertIs(torch.compiler, compiler)
        self.assertIs(compiler.torch, torch)
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
        self.assertEqual(
            function.__code__.co_varnames[:4],
            ("guard_entries", "keep_parameters", "keep_flags", "entry"),
        )
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

    def test_exports_copying_and_pickling_use_the_canonical_function(self):
        compiler = torch.compiler
        function = compiler.keep_tensor_guards_unsafe

        self.assertEqual(compiler.__all__, COMPILER_EXPORTS)
        namespace = {}
        exec("from torch_rs.compiler import *", namespace)
        self.assertEqual(
            {name for name in namespace if not name.startswith("__")},
            set(COMPILER_EXPORTS),
        )
        for name in COMPILER_EXPORTS:
            self.assertIs(namespace[name], getattr(compiler, name))

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
                lambda: function([], False, True),
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
                lambda: function([], keep_parameter=True),
                "keep_tensor_guards_unsafe() got an unexpected keyword argument "
                "'keep_parameter'",
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
                    entries = (
                        types.SimpleNamespace(
                            guard_type="TENSOR_MATCH", value=object()
                        ),
                        types.SimpleNamespace(guard_type="OTHER"),
                    )
                    self.assertEqual(
                        compiler.keep_tensor_guards_unsafe(iter(entries)),
                        [True, False],
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
            self.assertIs(compiler.torch, torch)
            self.assertIsNot(new_function, old_function)
            self.assertIsNot(compiler.__all__, old_exports)
            self.assertEqual(compiler.__all__, COMPILER_EXPORTS)
            self.assertIs(compiler.get_default_backend(), backend)
            entry = types.SimpleNamespace(guard_type="TENSOR_MATCH", value=object())
            self.assertEqual(new_function((entry,)), [True])
            self.assertIs(copy.copy(old_function), old_function)
            self.assertIs(copy.deepcopy(old_function), old_function)
            with self.assertRaises(pickle.PicklingError):
                pickle.dumps(old_function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                self.assertIs(
                    pickle.loads(pickle.dumps(new_function, protocol)), new_function
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

modules_before_call = set(sys.modules)
guard_entries = (
    types.SimpleNamespace(guard_type="TENSOR_MATCH", value=object()),
    types.SimpleNamespace(guard_type="OTHER"),
)
result = torch.compiler.keep_tensor_guards_unsafe(iter(guard_entries))
assert result == [True, False]
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
