import contextlib
import copy
import importlib
import inspect
import pickle
import pickletools
import sys
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
    "list_backends",
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


class _CountingIterator:
    def __init__(self, owner):
        self.owner = owner
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.owner.next_calls += 1
        if self.index == len(self.owner.tags):
            raise StopIteration
        tag = self.owner.tags[self.index]
        self.index += 1
        return tag


class _CountingIterable:
    def __init__(self, tags):
        self.tags = tags
        self.bool_calls = 0
        self.iter_calls = 0
        self.next_calls = 0

    def __bool__(self):
        self.bool_calls += 1
        return True

    def __iter__(self):
        self.iter_calls += 1
        if self.iter_calls != 1:
            raise AssertionError("exclude_tags was consumed more than once")
        return _CountingIterator(self)


class _FalseyIterable:
    def __init__(self):
        self.bool_calls = 0

    def __bool__(self):
        self.bool_calls += 1
        return False

    def __iter__(self):
        raise AssertionError("falsey exclude_tags should not be iterated")


class _BoolFailure:
    def __bool__(self):
        raise RuntimeError("exclude_tags truthiness failed")


class _IterFailure:
    def __iter__(self):
        raise LookupError("exclude_tags iteration failed")


class _InvalidIterator:
    def __iter__(self):
        return []


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CompilerListBackendsReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "compiler.list_backends differentials require pinned "
                "PyTorch 2.13.0"
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
        default = function()
        second_default = function()
        all_with_none = function(None)
        all_with_empty_tuple = function(())
        all_with_empty_list = function([])
        all_with_unknown = function(("unknown",))
        debug_filtered = function(("debug",))
        experimental_filtered = function(("experimental",))
        both_filtered = function(("debug", "experimental"))
        reversed_filtered = function(("experimental", "debug"))
        string_tags = function("debug")
        non_string_tags = function((1, object()))

        mutated = function()
        mutated.append("mutated")
        after_mutation = function()

        iterable = _CountingIterable(["debug", "experimental"])
        iterable_result = function(iterable)

        events = []

        def generated_tags():
            for tag in ("debug", "experimental"):
                events.append(("yield", tag))
                yield tag
            events.append(("finished", 2))

        generator = generated_tags()
        generator_result = function(generator)
        try:
            next(generator)
        except StopIteration:
            generator_exhausted = True
        else:
            generator_exhausted = False

        falsey = _FalseyIterable()
        falsey_result = function(falsey)

        results = (
            default,
            second_default,
            all_with_none,
            all_with_empty_tuple,
            all_with_empty_list,
            all_with_unknown,
            debug_filtered,
            experimental_filtered,
            both_filtered,
            reversed_filtered,
            string_tags,
            non_string_tags,
            after_mutation,
            iterable_result,
            generator_result,
            falsey_result,
        )
        return {
            "results": results,
            "result_types": tuple(type(result).__name__ for result in results),
            "element_types": tuple(
                tuple(type(backend).__name__ for backend in result)
                for result in results
            ),
            "fresh": (
                default is not second_default,
                default is not mutated,
                after_mutation is not mutated,
                all_with_none is not all_with_empty_tuple,
            ),
            "sorted": tuple(result == sorted(result) for result in results),
            "bool_calls": iterable.bool_calls,
            "iter_calls": iterable.iter_calls,
            "next_calls": iterable.next_calls,
            "events": events,
            "generator_exhausted": generator_exhausted,
            "falsey_bool_calls": falsey.bool_calls,
        }

    def exception_outcome(self, function, value):
        try:
            function(value)
        except BaseException as error:
            return type(error).__name__, str(error), error.args
        self.fail("expected an exception")

    def test_backend_lists_filtering_and_freshness_match_pytorch_2_13(self):
        self.assertEqual(
            self.behavior_outcome(torch.compiler.list_backends),
            self.behavior_outcome(reference_torch.compiler.list_backends),
        )

    def test_exclude_tag_iteration_and_argument_errors_match_pytorch_2_13(self):
        actual = torch.compiler.list_backends
        expected = reference_torch.compiler.list_backends

        for value in (_BoolFailure(), _IterFailure(), _InvalidIterator(), (["debug"],)):
            with self.subTest(value=type(value).__name__):
                self.assertEqual(
                    self.exception_outcome(actual, value),
                    self.exception_outcome(expected, value),
                )

        cases = (
            (lambda: actual((), ()), lambda: expected((), ())),
            (
                lambda: actual((), exclude_tags=()),
                lambda: expected((), exclude_tags=()),
            ),
            (lambda: actual(tags=()), lambda: expected(tags=())),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_signature_annotations_documentation_and_identity_match(self):
        actual_compiler = importlib.import_module("torch_rs.compiler")
        expected_compiler = importlib.import_module("torch.compiler")
        actual = actual_compiler.list_backends
        expected = expected_compiler.list_backends

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

    def test_direct_wildcard_imports_copy_and_pickle_match_pytorch_2_13(self):
        actual_compiler = torch.compiler
        expected_compiler = reference_torch.compiler
        actual = actual_compiler.list_backends
        expected = expected_compiler.list_backends

        for module, function in (
            (actual_compiler, actual),
            (expected_compiler, expected),
        ):
            namespace = {}
            exec(f"from {module.__name__} import list_backends", namespace)
            self.assertIs(namespace["list_backends"], function)

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
            torch.__all__.count("list_backends"),
            reference_torch.__all__.count("list_backends"),
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
            self.assertNotIn("list_backends", namespace)

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
            states = []

            for context in (contextlib.nullcontext(), module.no_grad()):
                with context:
                    before_grad = module.is_grad_enabled()
                    result = compiler.list_backends()
                    after_grad = module.is_grad_enabled()
                    states.append((before_grad, result, after_grad))

            return (
                states,
                compiler.get_default_backend() is backend,
                before_queries,
                (
                    compiler.is_compiling(),
                    compiler.is_dynamo_compiling(),
                    compiler.is_exporting(),
                ),
            )
        finally:
            compiler.set_default_backend(original_backend)

    def test_calls_preserve_compiler_and_grad_state_like_pytorch_2_13(self):
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
            old_function = compiler.list_backends
            old_exports = compiler.__all__
            reloaded = importlib.reload(compiler)
            new_function = reloaded.list_backends

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
                new_function(),
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

    def test_compile_and_registration_boundaries_remain_unsupported(self):
        self.assertTrue(callable(reference_torch.compile))
        self.assertTrue(callable(reference_torch.compiler.compile))
        self.assertTrue(callable(reference_torch.compiler.allow_in_graph))
        self.assertTrue(callable(reference_torch.compiler.substitute_in_graph))
        self.assertFalse(hasattr(reference_torch.compiler, "register_backend"))

        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch, "export"))
        self.assertFalse(hasattr(torch.compiler, "compile"))
        self.assertFalse(hasattr(torch.compiler, "allow_in_graph"))
        self.assertFalse(hasattr(torch.compiler, "substitute_in_graph"))
        self.assertFalse(hasattr(torch.compiler, "register_backend"))


if __name__ == "__main__":
    unittest.main()
