import copy
import importlib
import inspect
import pickle
import subprocess
import sys
import threading
import types
import unittest

import torch_rs as torch
from torch_rs import _compile_bytecode
from torch_rs import _compiler_state as _state


UNSUPPORTED_MESSAGE = (
    "torch.compile(): only backend='eager', fullgraph=True straight-line "
    "Tensor neg/abs/add functions with one or two positional exact native CPU "
    "float32 Tensor are supported; eager fallback, installed-PyTorch "
    "forwarding, callable backend invocation, CUDA compilation, and broader "
    "graph capture remain unsupported"
)


EAGER_COMPILE_MODEL_CALLS = []


def eager_compile_global_side_effect(value):
    EAGER_COMPILE_MODEL_CALLS.append("ran")
    return value + value


class TorchCompileEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.original_backend = torch.compiler.get_default_backend()
        self.registered_backends = dict(_state.registered_backends)
        self.registered_backend_fns = dict(_state.registered_backend_fns)
        torch.compiler.set_default_backend(None)
        _state.registered_backends.clear()
        _state.registered_backend_fns.clear()

    def tearDown(self):
        torch.compiler.set_default_backend(self.original_backend)
        _state.registered_backends.clear()
        _state.registered_backends.update(self.registered_backends)
        _state.registered_backend_fns.clear()
        _state.registered_backend_fns.update(self.registered_backend_fns)

    def test_signature_metadata_exports_copy_pickle_and_reload(self):
        function = torch.compile

        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "compile")
        self.assertEqual(function.__qualname__, "compile")
        self.assertEqual(function.__module__, "torch_rs")
        self.assertEqual(function.__defaults__, (None,))
        self.assertEqual(
            function.__kwdefaults__,
            {
                "fullgraph": False,
                "dynamic": None,
                "backend": None,
                "mode": None,
                "options": None,
                "name": None,
                "disable": False,
                "recompile_limit": None,
                "isolate_recompiles": False,
                "shapes_spec": None,
            },
        )
        self.assertEqual(function.__annotations__, {})
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(
            str(inspect.signature(function)),
            (
                "(model=None, *, fullgraph=False, dynamic=None, backend=None, "
                "mode=None, options=None, name=None, disable=False, "
                "recompile_limit=None, isolate_recompiles=False, "
                "shapes_spec=None)"
            ),
        )
        self.assertIn("argument binding", inspect.cleandoc(function.__doc__))
        self.assertIn("backend resolution", inspect.cleandoc(function.__doc__))
        self.assertIn("Tensor ``neg``", inspect.cleandoc(function.__doc__))
        self.assertIn("broader graph capture", inspect.cleandoc(function.__doc__))

        self.assertEqual(torch.__all__.count("compile"), 1)
        self.assertFalse(hasattr(torch._C, "compile"))
        self.assertNotIn("compile", torch._C.__all__)

        direct_import = {}
        exec("from torch_rs import compile", direct_import)
        self.assertIs(direct_import["compile"], function)

        wildcard_import = {}
        exec("from torch_rs import *", wildcard_import)
        self.assertIs(wildcard_import["compile"], function)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol)
                self.assertIn(b"torch_rs", payload)
                self.assertIn(b"compile", payload)
                self.assertIs(pickle.loads(payload), function)

        old_function = function
        self.assertIs(importlib.reload(torch), torch)
        self.assertIsNot(torch.compile, old_function)
        self.assertEqual(
            str(inspect.signature(torch.compile)),
            str(inspect.signature(old_function)),
        )
        with self.assertRaises(pickle.PicklingError) as raised:
            pickle.dumps(old_function)
        self.assertIn(
            "it's not the same object as torch_rs.compile",
            str(raised.exception),
        )

    def test_disable_true_returns_the_original_callable(self):
        calls = []

        def backend(graph_module, example_inputs):
            calls.append(("backend", graph_module, example_inputs))
            return graph_module.forward

        def model(value):
            calls.append(value)
            return value + 1

        self.assertIs(torch.compile(model, disable=True), model)
        self.assertEqual(model(2), 3)
        self.assertEqual(calls, [2])

        disabled_decorator = torch.compile(backend=backend, disable=True)
        self.assertIs(disabled_decorator(model), model)

        @torch.compile(disable=True, backend=backend)
        def decorated(value):
            calls.append(value)
            return value + 2

        self.assertEqual(decorated(3), 5)
        self.assertEqual(calls, [2, 3])

    def test_direct_and_decorator_forms_raise_before_user_or_backend_code_runs(self):
        model_calls = []
        backend_calls = []

        def backend(graph_module, example_inputs):
            backend_calls.append((graph_module, example_inputs))
            return graph_module.forward

        def model(*args, **kwargs):
            model_calls.append((args, kwargs))
            return "ran"

        shapes_spec = object()
        compiled = torch.compile(
            model,
            fullgraph=True,
            dynamic=False,
            backend=backend,
            mode="reduce-overhead",
            name="named_compile",
            recompile_limit=3,
            isolate_recompiles=True,
            shapes_spec=shapes_spec,
        )
        self.assertTrue(callable(compiled))
        self.assertIs(compiled.__wrapped__, model)
        self.assertIs(compiled._torch_rs_compile_backend, backend)
        self.assertIs(compiled._torch_rs_compile_fullgraph, True)
        self.assertIs(compiled._torch_rs_compile_dynamic, False)
        self.assertEqual(compiled._torch_rs_compile_mode, "reduce-overhead")
        self.assertIs(compiled._torch_rs_compile_options, None)
        self.assertEqual(compiled._torch_rs_compile_name, "named_compile")
        self.assertEqual(compiled._torch_rs_compile_recompile_limit, 3)
        self.assertIs(compiled._torch_rs_compile_isolate_recompiles, True)
        self.assertIs(compiled._torch_rs_compile_shapes_spec, shapes_spec)
        self.assertEqual(model_calls, [])
        self.assertEqual(backend_calls, [])
        with self.assertRaises(NotImplementedError) as raised:
            compiled("input", flag=True)
        self.assertEqual(str(raised.exception), UNSUPPORTED_MESSAGE)
        self.assertEqual(model_calls, [])
        self.assertEqual(backend_calls, [])

        options = {"trace": True}
        options_compiled = torch.compile(model, backend=backend, options=options)
        self.assertIs(options_compiled._torch_rs_compile_options, options)

        @torch.compile
        def bare_decorated(value):
            model_calls.append(("bare", value))
            return value

        with self.assertRaisesRegex(NotImplementedError, "^torch\\.compile\\(\\):"):
            bare_decorated(1)
        self.assertEqual(model_calls, [])

        configured = torch.compile(backend=backend)

        @configured
        def configured_decorated(value):
            model_calls.append(("configured", value))
            return value

        self.assertIs(configured_decorated._torch_rs_compile_backend, backend)
        with self.assertRaises(NotImplementedError) as raised:
            configured_decorated(2)
        self.assertEqual(str(raised.exception), UNSUPPORTED_MESSAGE)
        self.assertEqual(model_calls, [])
        self.assertEqual(backend_calls, [])

    def test_callable_backend_cannot_spoof_the_native_eager_path(self):
        model_calls = []
        backend_calls = []

        class EqualToEagerBackend:
            def __eq__(self, other):
                return other == "eager"

            def __call__(self, graph_module, example_inputs):
                backend_calls.append((graph_module, example_inputs))
                return graph_module.forward

        def model(value):
            model_calls.append(value)
            return value + value

        backend = EqualToEagerBackend()
        compiled = torch.compile(model, backend=backend, fullgraph=True)

        self.assertIs(compiled._torch_rs_compile_backend, backend)
        with self.assertRaises(NotImplementedError) as raised:
            compiled(torch.tensor([1.0], dtype=torch.float32))
        self.assertEqual(str(raised.exception), UNSUPPORTED_MESSAGE)
        self.assertEqual(model_calls, [])
        self.assertEqual(backend_calls, [])

    def test_eager_fullgraph_rejects_callable_object_without_metadata_side_effects(self):
        events = []

        class CallableObject:
            def __getattribute__(self, name):
                if name in (
                    "__annotations__",
                    "__dict__",
                    "__doc__",
                    "__module__",
                    "__name__",
                    "__qualname__",
                ):
                    events.append(name)
                return object.__getattribute__(self, name)

            def __call__(self, value):
                events.append("__call__")
                return value + value

        model = CallableObject()
        compiled = torch.compile(model, backend="eager", fullgraph=True)

        self.assertIs(compiled.__wrapped__, model)
        self.assertEqual(events, [])
        with self.assertRaises(NotImplementedError) as raised:
            compiled(torch.tensor([1.0], dtype=torch.float32))
        self.assertEqual(str(raised.exception), UNSUPPORTED_MESSAGE)
        self.assertEqual(events, [])

    def test_eager_fullgraph_executes_supported_tensor_programs_natively(self):
        def program(x):
            y = x.neg()
            return (y.abs() + x).add(x.neg())

        compiled = torch.compile(program, backend="eager", fullgraph=True)
        input = torch.tensor([[-2.0, 0.5, 3.0], [4.25, -5.5, 6.0]])
        expected = program(input)
        actual = compiled(input)

        self.assertIs(compiled.__wrapped__, program)
        self.assertIs(compiled._torch_rs_compile_backend, "eager")
        self.assertEqual(actual.tolist(), expected.tolist())
        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        self.assertEqual(actual.stride(), expected.stride())
        self.assertIs(actual.dtype, expected.dtype)
        self.assertEqual(actual.device, expected.device)

    def test_eager_fullgraph_executes_two_input_broadcasting_programs_natively(self):
        def matrix_vector(x, y):
            return x.neg().abs() + y.negative()

        def tensor_scalar(x, y):
            return (x + y).abs()

        def scalar_tensor(x, y):
            return x.add(y.neg())

        matrix = torch.tensor(
            [[-3.0, 0.5, 4.0], [2.25, -5.5, 6.75]],
            dtype=torch.float32,
            requires_grad=True,
        )
        vector = torch.tensor([1.0, -2.0, 0.25], dtype=torch.float32)
        scalar = torch.tensor(-1.25, dtype=torch.float32, requires_grad=True)
        other_matrix = torch.tensor(
            [[-0.5, 1.5, -2.5], [3.5, -4.5, 5.5]],
            dtype=torch.float32,
            requires_grad=True,
        )

        cases = (
            ("matrix_vector", matrix_vector, (matrix, vector)),
            ("tensor_scalar", tensor_scalar, (matrix, scalar)),
            ("scalar_tensor", scalar_tensor, (scalar, other_matrix)),
        )
        for case, program, inputs in cases:
            with self.subTest(case=case):
                compiled = torch.compile(program, backend="eager", fullgraph=True)
                expected = program(*inputs)
                actual = compiled(*inputs)

                self.assertEqual(actual.tolist(), expected.tolist())
                self.assertEqual(tuple(actual.shape), tuple(expected.shape))
                self.assertEqual(actual.stride(), expected.stride())
                self.assertIs(actual.dtype, expected.dtype)
                self.assertEqual(actual.device, expected.device)
                self.assertEqual(actual.requires_grad, expected.requires_grad)
                self.assertEqual(actual.storage_offset(), expected.storage_offset())
                self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
                for input in inputs:
                    if input.numel():
                        self.assertNotEqual(actual.data_ptr(), input.data_ptr())

    def test_eager_fullgraph_caches_graphs_by_code_and_input_metadata(self):
        def program(x):
            return x.neg().abs() + x

        original_lower = _compile_bytecode.lower_one_input_compile_graph
        calls = []

        def counting_lower(requested_program, input_metadata, *, name=None):
            calls.append((requested_program, input_metadata, name))
            return original_lower(requested_program, input_metadata, name=name)

        compiled = torch.compile(program, backend="eager", fullgraph=True)
        first = torch.tensor([[-2.0, 3.0], [4.0, -5.0]], dtype=torch.float32)
        second = torch.tensor([[1.0, -1.5], [2.5, -3.0]], dtype=torch.float32)
        different_shape = torch.tensor([1.5, -2.5, 3.5], dtype=torch.float32)

        try:
            _compile_bytecode.lower_one_input_compile_graph = counting_lower
            self.assertEqual(compiled(first).tolist(), program(first).tolist())
            self.assertEqual(compiled(second).tolist(), program(second).tolist())
            self.assertEqual(len(calls), 1)

            self.assertEqual(
                compiled(different_shape).tolist(),
                program(different_shape).tolist(),
            )
            self.assertEqual(len(calls), 2)
        finally:
            _compile_bytecode.lower_one_input_compile_graph = original_lower

    def test_eager_fullgraph_caches_two_input_graphs_by_all_input_metadata(self):
        def program(x, y):
            return x + y.abs()

        original_lower = _compile_bytecode.lower_compile_graph
        calls = []

        def counting_lower(requested_program, input_metadatas, *, name=None):
            calls.append((requested_program, input_metadatas, name))
            return original_lower(requested_program, input_metadatas, name=name)

        compiled = torch.compile(program, backend="eager", fullgraph=True)
        matrix = torch.tensor(
            [[-2.0, 3.0, 4.0], [5.0, -6.0, 7.0]],
            dtype=torch.float32,
        )
        same_shape_vector = torch.tensor([1.0, -1.5, 2.5], dtype=torch.float32)
        same_metadata_vector = torch.tensor([-3.0, 0.25, 4.0], dtype=torch.float32)
        strided_matrix = torch.tensor(
            [[-2.0, 5.0], [3.0, -6.0], [4.0, 7.0]],
            dtype=torch.float32,
        ).t()
        requires_grad_vector = torch.tensor(
            [1.0, -1.5, 2.5],
            dtype=torch.float32,
            requires_grad=True,
        )
        reshaped_vector = torch.tensor([[1.0, -1.5, 2.5]], dtype=torch.float32)
        changed_left = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=torch.float32,
        )

        try:
            _compile_bytecode.lower_compile_graph = counting_lower
            self.assertEqual(
                compiled(matrix, same_shape_vector).tolist(),
                program(matrix, same_shape_vector).tolist(),
            )
            self.assertEqual(
                compiled(matrix, same_metadata_vector).tolist(),
                program(matrix, same_metadata_vector).tolist(),
            )
            self.assertEqual(len(calls), 1)

            self.assertEqual(
                compiled(strided_matrix, same_shape_vector).tolist(),
                program(strided_matrix, same_shape_vector).tolist(),
            )
            self.assertEqual(len(calls), 2)

            self.assertEqual(
                compiled(matrix, requires_grad_vector).tolist(),
                program(matrix, requires_grad_vector).tolist(),
            )
            self.assertEqual(len(calls), 3)

            self.assertEqual(
                compiled(matrix, reshaped_vector).tolist(),
                program(matrix, reshaped_vector).tolist(),
            )
            self.assertEqual(len(calls), 4)

            self.assertEqual(
                compiled(changed_left, same_shape_vector).tolist(),
                program(changed_left, same_shape_vector).tolist(),
            )
            self.assertEqual(len(calls), 5)
        finally:
            _compile_bytecode.lower_compile_graph = original_lower

    def test_eager_fullgraph_recompile_limit_zero_rejects_first_graph(self):
        def program(x):
            return x.neg().abs()

        original_lower = _compile_bytecode.lower_one_input_compile_graph
        calls = []

        def counting_lower(requested_program, input_metadata, *, name=None):
            calls.append((requested_program, input_metadata, name))
            return original_lower(requested_program, input_metadata, name=name)

        compiled = torch.compile(
            program,
            backend="eager",
            fullgraph=True,
            recompile_limit=0,
        )
        try:
            _compile_bytecode.lower_one_input_compile_graph = counting_lower
            with self.assertRaisesRegex(NotImplementedError, "recompile_limit=0"):
                compiled(torch.tensor([1.0, -2.0], dtype=torch.float32))
            self.assertEqual(calls, [])
        finally:
            _compile_bytecode.lower_one_input_compile_graph = original_lower

    def test_eager_fullgraph_recompile_limit_one_rejects_shape_change(self):
        def program(x):
            return x.neg().abs() + x

        original_lower = _compile_bytecode.lower_one_input_compile_graph
        calls = []

        def counting_lower(requested_program, input_metadata, *, name=None):
            calls.append((requested_program, input_metadata, name))
            return original_lower(requested_program, input_metadata, name=name)

        compiled = torch.compile(
            program,
            backend="eager",
            fullgraph=True,
            recompile_limit=1,
        )
        first = torch.tensor([[-2.0, 3.0], [4.0, -5.0]], dtype=torch.float32)
        same_metadata = torch.tensor(
            [[1.0, -1.5], [2.5, -3.0]],
            dtype=torch.float32,
        )
        different_shape = torch.tensor([1.5, -2.5, 3.5], dtype=torch.float32)

        try:
            _compile_bytecode.lower_one_input_compile_graph = counting_lower
            self.assertEqual(compiled(first).tolist(), program(first).tolist())
            self.assertEqual(
                compiled(same_metadata).tolist(),
                program(same_metadata).tolist(),
            )
            self.assertEqual(len(calls), 1)

            with self.assertRaisesRegex(NotImplementedError, "recompile_limit=1"):
                compiled(different_shape)
            self.assertEqual(len(calls), 1)
            self.assertEqual(compiled(first).tolist(), program(first).tolist())
        finally:
            _compile_bytecode.lower_one_input_compile_graph = original_lower

    def test_eager_fullgraph_recompile_limit_one_is_atomic_for_concurrent_misses(self):
        def program(x):
            return x.neg().abs() + x

        original_lower = _compile_bytecode.lower_one_input_compile_graph
        calls = []
        calls_lock = threading.Lock()
        first_lower_started = threading.Event()
        second_lower_started = threading.Event()
        release_lowerers = threading.Event()

        def blocking_lower(requested_program, input_metadata, *, name=None):
            with calls_lock:
                calls.append((requested_program, input_metadata, name))
                is_first_call = len(calls) == 1
            if is_first_call:
                first_lower_started.set()
            else:
                second_lower_started.set()
            release_lowerers.wait(5.0)
            return original_lower(requested_program, input_metadata, name=name)

        compiled = torch.compile(
            program,
            backend="eager",
            fullgraph=True,
            recompile_limit=1,
        )
        first = torch.tensor([[-2.0, 3.0], [4.0, -5.0]], dtype=torch.float32)
        different_shape = torch.tensor([1.5, -2.5, 3.5], dtype=torch.float32)
        results = []
        results_lock = threading.Lock()

        def call_compiled(label, input):
            try:
                result = ("ok", label, compiled(input).tolist())
            except Exception as error:
                result = ("error", label, type(error), str(error))
            with results_lock:
                results.append(result)

        try:
            _compile_bytecode.lower_one_input_compile_graph = blocking_lower
            first_thread = threading.Thread(
                target=call_compiled,
                args=("first", first),
            )
            second_thread = threading.Thread(
                target=call_compiled,
                args=("second", different_shape),
            )

            first_thread.start()
            self.assertTrue(first_lower_started.wait(5.0))
            second_thread.start()
            second_lower_started.wait(0.25)
            release_lowerers.set()
            first_thread.join(5.0)
            second_thread.join(5.0)

            self.assertFalse(first_thread.is_alive())
            self.assertFalse(second_thread.is_alive())
            self.assertFalse(second_lower_started.is_set())
            self.assertEqual(len(calls), 1)
            self.assertCountEqual(
                [result[0] for result in results],
                ["ok", "error"],
            )
            self.assertIn(("ok", "first", program(first).tolist()), results)

            errors = [result for result in results if result[0] == "error"]
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0][1], "second")
            self.assertTrue(issubclass(errors[0][2], NotImplementedError))
            self.assertIn("recompile_limit=1", errors[0][3])
        finally:
            release_lowerers.set()
            _compile_bytecode.lower_one_input_compile_graph = original_lower

    def test_compiler_reset_clears_eager_fullgraph_compile_cache(self):
        def program(x):
            return x.neg().abs() + x

        original_lower = _compile_bytecode.lower_one_input_compile_graph
        calls = []

        def counting_lower(requested_program, input_metadata, *, name=None):
            calls.append((requested_program, input_metadata, name))
            return original_lower(requested_program, input_metadata, name=name)

        compiled = torch.compile(
            program,
            backend="eager",
            fullgraph=True,
            recompile_limit=1,
        )
        first = torch.tensor([[-2.0, 3.0], [4.0, -5.0]], dtype=torch.float32)
        different_shape = torch.tensor([1.5, -2.5, 3.5], dtype=torch.float32)

        try:
            _compile_bytecode.lower_one_input_compile_graph = counting_lower
            self.assertEqual(compiled(first).tolist(), program(first).tolist())
            with self.assertRaisesRegex(NotImplementedError, "recompile_limit=1"):
                compiled(different_shape)
            self.assertEqual(len(calls), 1)

            self.assertIs(torch.compiler.reset(), None)
            self.assertEqual(
                compiled(different_shape).tolist(),
                program(different_shape).tolist(),
            )
            self.assertEqual(len(calls), 2)
        finally:
            _compile_bytecode.lower_one_input_compile_graph = original_lower

    def test_eager_fullgraph_recompile_limit_one_rejects_second_input_change(self):
        def program(x, y):
            return x + y

        original_lower = _compile_bytecode.lower_compile_graph
        calls = []

        def counting_lower(requested_program, input_metadatas, *, name=None):
            calls.append((requested_program, input_metadatas, name))
            return original_lower(requested_program, input_metadatas, name=name)

        compiled = torch.compile(
            program,
            backend="eager",
            fullgraph=True,
            recompile_limit=1,
        )
        matrix = torch.tensor(
            [[-2.0, 3.0, 4.0], [5.0, -6.0, 7.0]],
            dtype=torch.float32,
        )
        vector = torch.tensor([1.0, -1.5, 2.5], dtype=torch.float32)
        reshaped_vector = torch.tensor([[1.0, -1.5, 2.5]], dtype=torch.float32)

        try:
            _compile_bytecode.lower_compile_graph = counting_lower
            self.assertEqual(
                compiled(matrix, vector).tolist(),
                program(matrix, vector).tolist(),
            )
            self.assertEqual(len(calls), 1)
            with self.assertRaisesRegex(NotImplementedError, "recompile_limit=1"):
                compiled(matrix, reshaped_vector)
            self.assertEqual(len(calls), 1)
            self.assertEqual(
                compiled(matrix, vector).tolist(),
                program(matrix, vector).tolist(),
            )
        finally:
            _compile_bytecode.lower_compile_graph = original_lower

    def test_eager_fullgraph_recompile_limit_rejects_invalid_values(self):
        def program(x):
            return x + x

        invalid_type_values = (True, False, "1", 1.0, object())
        for value in invalid_type_values:
            with self.subTest(value=repr(value)):
                with self.assertRaisesRegex(TypeError, "recompile_limit"):
                    torch.compile(
                        program,
                        backend="eager",
                        fullgraph=True,
                        recompile_limit=value,
                    )

        with self.assertRaisesRegex(ValueError, "recompile_limit"):
            torch.compile(
                program,
                backend="eager",
                fullgraph=True,
                recompile_limit=-1,
            )

    def test_eager_fullgraph_rejects_isolate_recompiles_control(self):
        calls = []

        def program(x):
            calls.append("program")
            return x + x

        compiled = torch.compile(
            program,
            backend="eager",
            fullgraph=True,
            isolate_recompiles=True,
        )
        with self.assertRaises(NotImplementedError) as raised:
            compiled(torch.tensor([1.0], dtype=torch.float32))
        self.assertEqual(str(raised.exception), UNSUPPORTED_MESSAGE)
        self.assertEqual(calls, [])

    def test_eager_fullgraph_cached_graphs_still_check_tensor_method_guards(self):
        def program(x):
            return x.neg()

        input = torch.tensor([2.0], dtype=torch.float32)
        compiled = torch.compile(program, backend="eager", fullgraph=True)
        self.assertEqual(compiled(input).tolist(), [-2.0])

        missing = object()
        original = torch.Tensor.__dict__.get("neg", missing)
        torch.Tensor.neg = lambda self: self + self
        try:
            with self.assertRaisesRegex(
                NotImplementedError,
                "patched Tensor operation bindings: .*Tensor\\.neg",
            ):
                compiled(input)
        finally:
            if original is missing:
                delattr(torch.Tensor, "neg")
            else:
                torch.Tensor.neg = original

    def test_eager_fullgraph_rejects_patched_tensor_operation_bindings(self):
        def call_neg(value):
            return value.neg()

        def call_negative(value):
            return value.negative()

        def call_abs(value):
            return value.abs()

        def call_absolute(value):
            return value.absolute()

        def call_add(value):
            return value.add(value)

        def call_dunder_add(value):
            return value + value

        def call_dunder_radd(value):
            return value.__radd__(value)

        def call_dunder_neg(value):
            return -value

        def call_dunder_abs(value):
            return value.__abs__()

        def patched_getattribute(self, name):
            if name == "neg":
                return lambda: self + self
            return object.__getattribute__(self, name)

        input = torch.tensor([2.0], dtype=torch.float32)
        cases = (
            ("neg", lambda self: self + self, call_neg, [4.0]),
            ("negative", lambda self: self + self, call_negative, [4.0]),
            ("abs", lambda self: self + self, call_abs, [4.0]),
            ("absolute", lambda self: self + self, call_absolute, [4.0]),
            ("add", lambda self, other: self.neg(), call_add, [-2.0]),
            ("__add__", lambda self, other: self.neg(), call_dunder_add, [-2.0]),
            (
                "__radd__",
                lambda self, other: self.neg(),
                call_dunder_radd,
                [-2.0],
            ),
            ("__neg__", lambda self: self + self, call_dunder_neg, [4.0]),
            ("__abs__", lambda self: self + self, call_dunder_abs, [4.0]),
            (
                "__getattribute__",
                patched_getattribute,
                call_neg,
                [4.0],
            ),
        )

        missing = object()
        for name, replacement, program, expected_eager in cases:
            with self.subTest(binding=name):
                original = torch.Tensor.__dict__.get(name, missing)
                setattr(torch.Tensor, name, replacement)
                try:
                    self.assertEqual(program(input).tolist(), expected_eager)
                    compiled = torch.compile(
                        program,
                        backend="eager",
                        fullgraph=True,
                    )
                    with self.assertRaisesRegex(
                        NotImplementedError,
                        f"patched Tensor operation bindings: .*Tensor\\.{name}",
                    ):
                        compiled(input)
                finally:
                    if original is missing:
                        delattr(torch.Tensor, name)
                    else:
                        setattr(torch.Tensor, name, original)

    def test_eager_fullgraph_relowers_for_runtime_input_metadata(self):
        def program(x):
            return x.neg().abs() + x

        compiled = torch.compile(program, backend="eager", fullgraph=True)
        first = torch.tensor([[-2.0, 3.0], [4.0, -5.0]], dtype=torch.float32)
        second = torch.tensor([1.5, -2.5, 3.5], dtype=torch.float32)

        first_actual = compiled(first)
        second_actual = compiled(second)

        self.assertEqual(first_actual.tolist(), program(first).tolist())
        self.assertEqual(tuple(first_actual.shape), (2, 2))
        self.assertEqual(second_actual.tolist(), program(second).tolist())
        self.assertEqual(tuple(second_actual.shape), (3,))

    def test_eager_fullgraph_rejects_unsupported_programs_without_running_them(self):
        calls = []

        def closure_factory():
            flag = True

            def closure(value):
                calls.append("closure")
                if flag:
                    return value + value
                return value

            return closure

        def global_call(value):
            return torch.abs(value)

        def control_flow(value):
            if value:
                return value
            return value.neg()

        def exception_handling(value):
            try:
                return value + value
            except Exception:
                return value

        def mutation(value):
            value += value
            return value

        def unsupported_method(value):
            return value.relu()

        input = torch.tensor([1.0, -2.0], dtype=torch.float32)
        cases = (
            ("closure", closure_factory(), "closures"),
            ("global", global_call, "global or import access"),
            (
                "global side effect",
                eager_compile_global_side_effect,
                "global or import access",
            ),
            ("control flow", control_flow, "control flow"),
            ("exception handling", exception_handling, "exception handling"),
            ("mutation", mutation, "mutation"),
            ("unsupported method", unsupported_method, "Tensor.relu"),
        )
        EAGER_COMPILE_MODEL_CALLS.clear()
        for case, program, message in cases:
            with self.subTest(case=case):
                compiled = torch.compile(program, backend="eager", fullgraph=True)
                with self.assertRaisesRegex(NotImplementedError, message):
                    compiled(input)
                self.assertEqual(calls, [])
                self.assertEqual(EAGER_COMPILE_MODEL_CALLS, [])

    def test_eager_fullgraph_rejects_two_input_unsupported_forms(self):
        calls = []

        def program(x, y):
            calls.append("program")
            return x + y

        def scalar_operand(x, y):
            return x + 1.0

        input = torch.tensor([1.0, -2.0], dtype=torch.float32)
        compiled = torch.compile(program, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(
            NotImplementedError,
            "one or two positional Tensor arguments",
        ):
            compiled(input, input, input)
        self.assertEqual(calls, [])

        with self.assertRaisesRegex(TypeError, "expected exact native"):
            compiled(input, 1.0)
        self.assertEqual(calls, [])

        compiled_scalar_operand = torch.compile(
            scalar_operand,
            backend="eager",
            fullgraph=True,
        )
        with self.assertRaisesRegex(NotImplementedError, "non-Tensor right operand"):
            compiled_scalar_operand(input, input)
        self.assertEqual(calls, [])

    def test_eager_fullgraph_rejects_kwargs_without_running_model(self):
        calls = []

        def program(value):
            calls.append("program")
            return value + value

        compiled = torch.compile(program, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(NotImplementedError, "keyword arguments: value"):
            compiled(value=torch.tensor([1.0], dtype=torch.float32))
        self.assertEqual(calls, [])

    def test_backend_none_resolves_default_and_registered_backend_names(self):
        backend_calls = []

        def model(value):
            raise AssertionError(f"model should not run: {value!r}")

        def registered_backend(graph_module, example_inputs):
            backend_calls.append(("registered", graph_module, example_inputs))
            return graph_module.forward

        def default_backend(graph_module, example_inputs):
            backend_calls.append(("default", graph_module, example_inputs))
            return graph_module.forward

        torch.compiler.register_backend(registered_backend, name="zz_compile")
        torch.compiler.register_backend(default_backend, name="zz_default")

        compiled = torch.compile(model, backend="zz_compile")
        self.assertIs(compiled._torch_rs_compile_backend, registered_backend)
        with self.assertRaises(NotImplementedError):
            compiled("value")
        self.assertEqual(backend_calls, [])

        torch.compiler.set_default_backend("zz_default")
        default_compiled = torch.compile(model)
        self.assertIs(default_compiled._torch_rs_compile_backend, default_backend)
        with self.assertRaises(NotImplementedError):
            default_compiled("value")
        self.assertEqual(backend_calls, [])

        torch.compiler.set_default_backend(registered_backend)
        callable_default_compiled = torch.compile(model)
        self.assertIs(
            callable_default_compiled._torch_rs_compile_backend,
            registered_backend,
        )
        with self.assertRaises(NotImplementedError):
            callable_default_compiled("value")
        self.assertEqual(backend_calls, [])

    def test_decorator_form_snapshots_default_backend_at_factory_time(self):
        def model(value):
            raise AssertionError(f"model should not run: {value!r}")

        def first_backend(graph_module, example_inputs):
            raise AssertionError("first backend should not be invoked")

        def second_backend(graph_module, example_inputs):
            raise AssertionError("second backend should not be invoked")

        torch.compiler.register_backend(first_backend, name="zz_first")
        torch.compiler.register_backend(second_backend, name="zz_second")

        torch.compiler.set_default_backend("zz_first")
        configured = torch.compile()
        torch.compiler.set_default_backend("zz_second")
        compiled = configured(model)
        self.assertIs(compiled._torch_rs_compile_backend, first_backend)

        torch.compiler.set_default_backend("missing_at_factory")
        invalid_configured = torch.compile()
        torch.compiler.set_default_backend("zz_second")
        with self.assertRaises(RuntimeError) as raised:
            invalid_configured(model)
        self.assertIn("Invalid backend: 'missing_at_factory'", str(raised.exception))

    def test_invalid_backend_names_and_types_fail_without_invocation(self):
        calls = []

        def model():
            calls.append("model")

        def backend(graph_module, example_inputs):
            calls.append("backend")
            return graph_module.forward

        torch.compiler.register_backend(backend, name="zz_valid")

        with self.assertRaises(RuntimeError) as raised:
            torch.compile(model, backend="missing")
        self.assertEqual(
            str(raised.exception),
            (
                "Invalid backend: 'missing'. Available backend names are: "
                "'eager', 'inductor', 'zz_valid'"
            ),
        )
        self.assertEqual(calls, [])

        torch.compiler.set_default_backend("missing_default")
        with self.assertRaises(RuntimeError) as raised:
            torch.compile(model)
        self.assertIn("Invalid backend: 'missing_default'", str(raised.exception))
        self.assertEqual(calls, [])

        missing_decorator = torch.compile(backend="missing")
        with self.assertRaises(RuntimeError) as raised:
            missing_decorator(model)
        self.assertIn("Invalid backend: 'missing'", str(raised.exception))
        self.assertEqual(calls, [])

        disabled_missing_decorator = torch.compile(backend="missing", disable=True)
        with self.assertRaises(RuntimeError) as raised:
            disabled_missing_decorator(model)
        self.assertIn("Invalid backend: 'missing'", str(raised.exception))
        self.assertEqual(calls, [])

        with self.assertRaises(TypeError) as raised:
            torch.compile(model, backend=object())
        self.assertEqual(
            str(raised.exception),
            "backend must be a string or callable, got <class 'object'>",
        )
        self.assertEqual(calls, [])

    def test_invalid_bound_models_and_configurations_fail_before_execution(self):
        calls = []

        def model(value):
            calls.append(value)
            return value

        with self.assertRaises(RuntimeError) as raised:
            torch.compile()(None)
        self.assertEqual(str(raised.exception), "Model can't be None")

        with self.assertRaises(AssertionError) as raised:
            torch.compile(123, disable=True)
        self.assertEqual(
            str(raised.exception),
            "A callable function is expected, but <class 'int'> is provided.",
        )

        with self.assertRaises(RuntimeError) as raised:
            torch.compile(model, mode="default", options={})
        self.assertEqual(
            str(raised.exception),
            (
                "Either mode or options can be specified, but both can't be "
                "specified at the same time."
            ),
        )
        self.assertEqual(calls, [])

        configured = torch.compile(mode="default", options={}, disable=True)
        with self.assertRaises(RuntimeError) as raised:
            configured(model)
        self.assertEqual(
            str(raised.exception),
            (
                "Either mode or options can be specified, but both can't be "
                "specified at the same time."
            ),
        )
        self.assertEqual(calls, [])

    def test_importing_and_calling_compile_shell_does_not_import_pytorch(self):
        script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

calls = []

def backend(graph_module, example_inputs):
    calls.append("backend")
    return graph_module.forward

def model(value):
    calls.append(("model", value))
    return value

modules_before_call = set(sys.modules)
compiled = torch.compile(model, backend=backend)
try:
    compiled("value")
except NotImplementedError as error:
    assert "graph capture" in str(error)
else:
    raise AssertionError("compiled shell should raise")

assert calls == []
assert torch.compile(disable=True)(model) is model
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
