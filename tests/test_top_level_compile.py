import copy
import importlib
import inspect
import pickle
import subprocess
import sys
import types
import unittest

import torch_rs as torch
from torch_rs import _compile_trace
from torch_rs import _compiler_state as _state


UNSUPPORTED_MESSAGE = (
    "torch.compile(): only argument binding, disable=True pass-through, "
    "backend resolution, and the built-in eager fullgraph CPU float32 "
    "Tensor.neg().abs() graphlet are implemented; broader graph capture, "
    "graph execution, eager fallback, installed-PyTorch forwarding, and "
    "backend invocation are not supported"
)


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

    def assert_native_tensor_matches(self, actual, expected):
        self.assertIsInstance(actual, torch.Tensor)
        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        self.assertEqual(actual.stride(), expected.stride())
        self.assertIs(actual.dtype, expected.dtype)
        self.assertEqual(actual.device, expected.device)
        self.assertEqual(actual.storage_offset(), expected.storage_offset())
        self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
        self.assertEqual(actual.requires_grad, expected.requires_grad)
        self.assertEqual(actual.tolist(), expected.tolist())

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
        self.assertIn("graph execution", inspect.cleandoc(function.__doc__))

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

    def test_eager_fullgraph_abs_neg_executes_private_trace_graph(self):
        execute_calls = []

        def model(x):
            return x.neg().abs()

        original_execute = _compile_trace.execute_compile_trace_graph

        def recording_execute(graph, input):
            execute_calls.append((graph, input))
            return original_execute(graph, input)

        input = torch.tensor(
            [[-3.25, -0.0, 1.5], [2.0, -4.5, 0.25]],
            dtype=torch.float32,
        )
        expected = input.neg().abs()
        compiled = torch.compile(model, backend="eager", fullgraph=True)

        _compile_trace.execute_compile_trace_graph = recording_execute
        try:
            actual = compiled(input)
            second_actual = compiled(input)
        finally:
            _compile_trace.execute_compile_trace_graph = original_execute

        self.assert_native_tensor_matches(actual, expected)
        self.assert_native_tensor_matches(second_actual, expected)
        self.assertEqual(len(execute_calls), 2)
        graph, executed_input = execute_calls[0]
        self.assertIs(executed_input, input)
        second_graph, second_input = execute_calls[1]
        self.assertIs(compiled._torch_rs_compile_trace_graph, second_graph)
        self.assertIsNot(second_graph, graph)
        self.assertIs(second_input, input)
        self.assertEqual(graph.name, "model")
        self.assertEqual(
            [operation.target for operation in graph.operations],
            ["neg", "abs"],
        )
        self.assertEqual(graph.operations[0].inputs, ("arg0",))
        self.assertEqual(graph.operations[1].inputs, ("neg_0",))
        self.assertEqual(graph.output, "abs_1")

    def test_eager_fullgraph_abs_neg_rejects_nearby_public_variants(self):
        calls = []

        def abs_neg(x):
            return x.neg().abs()

        class CallableModel:
            def __call__(self, x):
                calls.append(("callable", x))
                return x.neg().abs()

        def self_add(x):
            calls.append(("self_add", x))
            return x + x

        class EagerEqualBackend:
            def __eq__(self, other):
                return other == "eager"

            def __call__(self, graph_module, example_inputs):
                calls.append(("backend", graph_module, example_inputs))
                return graph_module.forward

        input = torch.tensor([[-1.0, 2.0]], dtype=torch.float32)

        unsupported_before_trace = (
            torch.compile(abs_neg, backend="eager"),
            torch.compile(abs_neg, backend="eager", fullgraph=1),
            torch.compile(abs_neg, backend="eager", fullgraph=True, dynamic=False),
            torch.compile(abs_neg, backend="eager", fullgraph=True, mode="default"),
            torch.compile(abs_neg, backend="eager", fullgraph=True, options={}),
            torch.compile(abs_neg, backend="eager", fullgraph=True, name="named"),
            torch.compile(
                abs_neg,
                backend="eager",
                fullgraph=True,
                recompile_limit=1,
            ),
            torch.compile(
                abs_neg,
                backend="eager",
                fullgraph=True,
                isolate_recompiles=True,
            ),
            torch.compile(
                abs_neg,
                backend="eager",
                fullgraph=True,
                shapes_spec=object(),
            ),
            torch.compile(CallableModel(), backend="eager", fullgraph=True),
            torch.compile(abs_neg, backend=EagerEqualBackend(), fullgraph=True),
            torch.compile(abs_neg, fullgraph=True),
            torch.compile(abs_neg, backend="inductor", fullgraph=True),
        )
        for compiled in unsupported_before_trace:
            with self.subTest(compiled=compiled):
                with self.assertRaises(NotImplementedError) as raised:
                    compiled(input)
                self.assertEqual(str(raised.exception), UNSUPPORTED_MESSAGE)
        self.assertEqual(calls, [])

        compiled = torch.compile(abs_neg, backend="eager", fullgraph=True)
        for args, kwargs in (
            ((), {}),
            ((input, input), {}),
            ((input,), {"scale": 1}),
            ((), {"x": input}),
        ):
            with self.subTest(args=args, kwargs=kwargs):
                with self.assertRaises(NotImplementedError) as raised:
                    compiled(*args, **kwargs)
                self.assertEqual(str(raised.exception), UNSUPPORTED_MESSAGE)
        self.assertEqual(calls, [])

        compiled_self_add = torch.compile(self_add, backend="eager", fullgraph=True)
        with self.assertRaises(NotImplementedError) as raised:
            compiled_self_add(input)
        self.assertEqual(str(raised.exception), UNSUPPORTED_MESSAGE)
        self.assertEqual(calls, [])

    def test_eager_fullgraph_abs_neg_rejects_proxy_dependent_functions(self):
        calls = []

        def proxy_dependent(x):
            calls.append(x)
            if type(x) is torch.Tensor:
                return x + x
            return x.neg().abs()

        input = torch.tensor([[-2.0, 3.0]], dtype=torch.float32)
        compiled = torch.compile(proxy_dependent, backend="eager", fullgraph=True)

        with self.assertRaises(NotImplementedError) as raised:
            compiled(input)

        self.assertEqual(str(raised.exception), UNSUPPORTED_MESSAGE)
        self.assertEqual(calls, [])

    def test_eager_fullgraph_abs_neg_rejects_stateful_functions_after_reset(self):
        calls = []
        use_abs_neg = True

        def stateful(x):
            calls.append(x)
            if use_abs_neg:
                return x.neg().abs()
            return x + x

        input = torch.tensor([[-2.0, 3.0]], dtype=torch.float32)
        compiled = torch.compile(stateful, backend="eager", fullgraph=True)

        with self.assertRaises(NotImplementedError) as raised:
            compiled(input)
        self.assertEqual(str(raised.exception), UNSUPPORTED_MESSAGE)

        use_abs_neg = False
        torch.compiler.reset()

        with self.assertRaises(NotImplementedError) as raised:
            compiled(input)
        self.assertEqual(str(raised.exception), UNSUPPORTED_MESSAGE)
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

def abs_neg_model(x):
    return x.neg().abs()

input = torch.tensor([[-3.0, 0.0, 2.5]], dtype=torch.float32)
expected = input.neg().abs()
compiled = torch.compile(abs_neg_model, backend="eager", fullgraph=True)
actual = compiled(input)
assert actual.tolist() == expected.tolist()
assert actual.shape == expected.shape
assert actual.stride() == expected.stride()
assert actual.dtype is expected.dtype
assert actual.device == expected.device
assert actual.requires_grad is expected.requires_grad
assert [
    operation.target
    for operation in compiled._torch_rs_compile_trace_graph.operations
] == ["neg", "abs"]
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
