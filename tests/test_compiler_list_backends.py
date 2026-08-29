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


FUNCTION_DOC = '''Return valid strings that can be passed to `torch.compile(..., backend="name")`.

    Args:
        exclude_tags(optional): A tuple of strings representing tags to exclude.
    '''

ALL_BACKENDS = [
    "aot_eager",
    "aot_eager_decomp_partition",
    "aot_eager_decomp_partition_crossref",
    "aot_eager_decomp_partition_with_mode",
    "aot_eager_default_partitioner",
    "aot_ts",
    "cudagraphs",
    "dynamo_accuracy_minifier_backend",
    "dynamo_minifier_backend",
    "eager",
    "eager_debug",
    "eager_noexcept",
    "inductor",
    "invoke_subgraph",
    "non_leaf_compile_error_TESTING_ONLY",
    "openxla",
    "openxla_eval",
    "pre_dispatch_eager",
    "relu_accuracy_error_TESTING_ONLY",
    "relu_compile_error_TESTING_ONLY",
    "relu_runtime_error_TESTING_ONLY",
    "ts",
    "tvm",
]
DEFAULT_BACKENDS = ["cudagraphs", "inductor", "openxla", "tvm"]
DEBUG_FILTERED_BACKENDS = [
    "cudagraphs",
    "inductor",
    "openxla",
    "openxla_eval",
    "tvm",
]
EXPERIMENTAL_FILTERED_BACKENDS = [
    name for name in ALL_BACKENDS if name != "openxla_eval"
]
COMPILER_EXPORTS = [
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
]


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


class CompilerListBackendsTests(unittest.TestCase):
    def assert_backend_list(self, actual, expected):
        self.assertIs(type(actual), list)
        self.assertEqual(actual, expected)
        self.assertEqual(actual, sorted(actual))
        for backend in actual:
            self.assertIs(type(backend), str)

    def test_default_result_and_exclude_tag_filtering(self):
        function = torch.compiler.list_backends

        self.assert_backend_list(function(), DEFAULT_BACKENDS)
        self.assert_backend_list(function(("debug", "experimental")), DEFAULT_BACKENDS)
        self.assert_backend_list(function(("experimental", "debug")), DEFAULT_BACKENDS)
        self.assert_backend_list(function(("debug",)), DEBUG_FILTERED_BACKENDS)
        self.assert_backend_list(
            function(("experimental",)), EXPERIMENTAL_FILTERED_BACKENDS
        )

        for exclude_tags in (None, (), [], {}, set(), False, 0, "", ("unknown",)):
            with self.subTest(exclude_tags=exclude_tags):
                self.assert_backend_list(function(exclude_tags), ALL_BACKENDS)

        self.assert_backend_list(function(("debug", "unknown")), DEBUG_FILTERED_BACKENDS)
        self.assert_backend_list(function(("eager",)), ALL_BACKENDS)
        self.assert_backend_list(function((1, object())), ALL_BACKENDS)
        self.assert_backend_list(function("debug"), ALL_BACKENDS)

    def test_repeated_calls_return_fresh_lists(self):
        function = torch.compiler.list_backends

        first = function()
        second = function()
        self.assert_backend_list(first, DEFAULT_BACKENDS)
        self.assert_backend_list(second, DEFAULT_BACKENDS)
        self.assertIsNot(first, second)

        first.append("mutated")
        self.assert_backend_list(function(), DEFAULT_BACKENDS)

        first_all = function(None)
        second_all = function(())
        self.assert_backend_list(first_all, ALL_BACKENDS)
        self.assert_backend_list(second_all, ALL_BACKENDS)
        self.assertIsNot(first_all, second_all)

    def test_exclude_tags_materialization_matches_pytorch_2_13(self):
        iterable = _CountingIterable(["debug", "experimental"])
        self.assert_backend_list(torch.compiler.list_backends(iterable), DEFAULT_BACKENDS)
        self.assertEqual(iterable.bool_calls, 1)
        self.assertEqual(iterable.iter_calls, 1)
        self.assertEqual(iterable.next_calls, 3)

        events = []

        def generated_tags():
            for tag in ("debug", "experimental"):
                events.append(("yield", tag))
                yield tag
            events.append(("finished", 2))

        generator = generated_tags()
        self.assert_backend_list(torch.compiler.list_backends(generator), DEFAULT_BACKENDS)
        self.assertEqual(
            events,
            [("yield", "debug"), ("yield", "experimental"), ("finished", 2)],
        )
        with self.assertRaises(StopIteration):
            next(generator)

        falsey = _FalseyIterable()
        self.assert_backend_list(torch.compiler.list_backends(falsey), ALL_BACKENDS)
        self.assertEqual(falsey.bool_calls, 1)

    def test_exclude_tags_errors_propagate(self):
        function = torch.compiler.list_backends

        with self.assertRaises(RuntimeError) as bool_raised:
            function(_BoolFailure())
        self.assertEqual(str(bool_raised.exception), "exclude_tags truthiness failed")

        with self.assertRaises(LookupError) as iter_raised:
            function(_IterFailure())
        self.assertEqual(str(iter_raised.exception), "exclude_tags iteration failed")

        with self.assertRaises(TypeError) as invalid_iterator_raised:
            function(_InvalidIterator())
        invalid_iterator_message = "iter() returned non-iterator of type 'list'"
        self.assertEqual(
            str(invalid_iterator_raised.exception), invalid_iterator_message
        )
        self.assertEqual(
            invalid_iterator_raised.exception.args, (invalid_iterator_message,)
        )

        with self.assertRaises(TypeError) as unhashable_raised:
            function((["debug"],))
        unhashable_message = "unhashable type: 'list'"
        self.assertEqual(str(unhashable_raised.exception), unhashable_message)
        self.assertEqual(unhashable_raised.exception.args, (unhashable_message,))

    def test_signature_annotations_documentation_and_module_identity(self):
        compiler = importlib.import_module("torch_rs.compiler")
        function = compiler.list_backends

        self.assertIs(torch.compiler, compiler)
        self.assertIs(sys.modules["torch_rs.compiler"], compiler)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(
            inspect.signature(function),
            inspect.Signature(
                [
                    inspect.Parameter(
                        "exclude_tags",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=("debug", "experimental"),
                    )
                ],
                return_annotation=list[str],
            ),
        )
        self.assertEqual(function.__annotations__, {"return": list[str]})
        self.assertEqual(typing.get_type_hints(function), {"return": list[str]})
        self.assertEqual(function.__name__, "list_backends")
        self.assertEqual(function.__qualname__, "list_backends")
        self.assertEqual(function.__module__, "torch_rs.compiler")
        self.assertIs(inspect.getmodule(function), compiler)
        self.assertEqual(
            inspect.cleandoc(function.__doc__),
            inspect.cleandoc(FUNCTION_DOC),
        )
        self.assertEqual(function.__defaults__, (("debug", "experimental"),))
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_direct_wildcard_imports_copy_and_pickle_use_the_canonical_module(self):
        compiler = torch.compiler
        function = compiler.list_backends

        direct_namespace = {}
        exec("from torch_rs.compiler import list_backends", direct_namespace)
        self.assertIs(direct_namespace["list_backends"], function)

        self.assertEqual(compiler.__all__, COMPILER_EXPORTS)
        wildcard_namespace = {}
        exec("from torch_rs.compiler import *", wildcard_namespace)
        self.assertEqual(
            {name for name in wildcard_namespace if not name.startswith("__")},
            set(COMPILER_EXPORTS),
        )
        for name in COMPILER_EXPORTS:
            self.assertIs(wildcard_namespace[name], getattr(compiler, name))

        self.assertNotIn("compiler", torch.__all__)
        self.assertNotIn("list_backends", torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("compiler", top_level_namespace)
        self.assertNotIn("list_backends", top_level_namespace)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.compiler", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_argument_errors_match_pytorch_2_13(self):
        function = torch.compiler.list_backends
        cases = (
            (
                lambda: function((), ()),
                "list_backends() takes from 0 to 1 positional arguments but 2 "
                "were given",
            ),
            (
                lambda: function((), exclude_tags=()),
                "list_backends() got multiple values for argument 'exclude_tags'",
            ),
            (
                lambda: function(tags=()),
                "list_backends() got an unexpected keyword argument 'tags'",
            ),
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
        original_guard_collectives = compiler.set_enable_guard_collectives(False)

        def backend(graph_module, example_inputs):
            return graph_module.forward

        try:
            compiler.set_default_backend(backend)
            self.assertIs(compiler.set_enable_guard_collectives(True), False)
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
                    self.assert_backend_list(compiler.list_backends(), DEFAULT_BACKENDS)
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

            self.assertIs(compiler.set_enable_guard_collectives(False), True)

            old_function = compiler.list_backends
            old_exports = compiler.__all__
            reloaded = importlib.reload(compiler)
            new_function = reloaded.list_backends

            self.assertIs(reloaded, compiler)
            self.assertIs(torch.compiler, compiler)
            self.assertIs(sys.modules["torch_rs.compiler"], compiler)
            self.assertIsNot(new_function, old_function)
            self.assertIsNot(compiler.__all__, old_exports)
            self.assertEqual(compiler.__all__, COMPILER_EXPORTS)
            self.assertIs(compiler.get_default_backend(), backend)
            self.assert_backend_list(new_function(), DEFAULT_BACKENDS)

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
            compiler.set_enable_guard_collectives(original_guard_collectives)

    def test_compilation_execution_and_registration_apis_remain_unsupported(self):
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch, "export"))
        self.assertFalse(hasattr(torch.compiler, "compile"))
        self.assertFalse(hasattr(torch.compiler, "allow_in_graph"))
        self.assertFalse(hasattr(torch.compiler, "substitute_in_graph"))
        self.assertFalse(hasattr(torch.compiler, "register_backend"))

    def test_importing_and_calling_does_not_import_pytorch_or_a_registry(self):
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
assert torch.compiler.list_backends() == ["cudagraphs", "inductor", "openxla", "tvm"]
assert torch.compiler.list_backends(None)[0] == "aot_eager"
assert "openxla_eval" in torch.compiler.list_backends(("debug",))
assert set(sys.modules) == modules_before_call
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
assert not any(
    name.startswith("torch_rs._dynamo")
    or name.startswith("torch_rs.compiler.backends")
    or name == "torch_rs.compiler.registry"
    for name in sys.modules
)
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
