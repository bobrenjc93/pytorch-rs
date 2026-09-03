import copy
import functools
import importlib
import inspect
import pickle
import subprocess
import sys
import types
import unittest

import torch_rs as torch
from torch_rs import _compiler_state as _state


UNSUPPORTED_MESSAGE = (
    "torch.compile(): graph capture/execution is supported only for exact CPU "
    "float32 unary relu functions; eager fallback, installed-PyTorch forwarding, "
    "backend invocation, and CUDA compilation are not supported"
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

    def test_unary_relu_graphlet_executes_natively_with_runtime_scope_guards(self):
        def model(x):
            return x.relu()

        compiled = torch.compile(model, backend="eager")
        self.assertIs(compiled.__wrapped__, model)
        self.assertEqual(compiled._torch_rs_compile_graph, "relu")
        self.assertEqual(compiled._torch_rs_compile_execution, "torch_rs")
        self.assertEqual(compiled(torch.tensor([-1.0, 2.0])).tolist(), [0.0, 2.0])

        with self.assertRaises(NotImplementedError) as raised:
            compiled(torch.tensor([1.0], requires_grad=True))
        self.assertEqual(str(raised.exception), UNSUPPORTED_MESSAGE)

        self.assertEqual(compiled(x=torch.tensor([-1.0, 2.0])).tolist(), [0.0, 2.0])

        with self.assertRaises(NotImplementedError) as raised:
            compiled("input")
        self.assertEqual(str(raised.exception), UNSUPPORTED_MESSAGE)

        lambda_compiled = torch.compile(lambda x: x.relu(), backend="eager")
        self.assertEqual(
            lambda_compiled(torch.tensor([-2.0, 3.0])).tolist(),
            [0.0, 3.0],
        )

    def test_wrapped_relu_functions_are_not_miscompiled_or_executed(self):
        calls = []

        def original(x):
            return x.relu()

        @functools.wraps(original)
        def wrapped(x):
            calls.append("wrapped")
            return x.relu()

        wrapped_compiled = torch.compile(wrapped, backend="eager")
        self.assertFalse(hasattr(wrapped_compiled, "_torch_rs_compile_graph"))
        self.assertEqual(calls, [])
        with self.assertRaises(NotImplementedError) as raised:
            wrapped_compiled(torch.tensor([-1.0, 2.0]))
        self.assertEqual(str(raised.exception), UNSUPPORTED_MESSAGE)
        self.assertEqual(calls, [])

        @functools.wraps(original)
        def wrapped_other_method(x):
            calls.append("wrapped_other_method")
            return x.sqrt()

        other_compiled = torch.compile(wrapped_other_method, backend="eager")
        self.assertFalse(hasattr(other_compiled, "_torch_rs_compile_graph"))
        with self.assertRaises(NotImplementedError) as raised:
            other_compiled(torch.tensor([4.0]))
        self.assertEqual(str(raised.exception), UNSUPPORTED_MESSAGE)
        self.assertEqual(calls, [])

    def test_top_level_relu_graphlet_guards_the_public_relu_binding(self):
        original_relu = torch.relu

        def model(x):
            return torch.relu(x)

        compiled = torch.compile(model, backend="eager")
        self.assertEqual(compiled._torch_rs_compile_graph, "relu")
        self.assertEqual(compiled(torch.tensor([-1.0, 2.0])).tolist(), [0.0, 2.0])

        def replacement_relu(input):
            return input

        try:
            torch.relu = replacement_relu
            rebound_compiled = torch.compile(model, backend="eager")
            self.assertFalse(hasattr(rebound_compiled, "_torch_rs_compile_graph"))
            with self.assertRaises(NotImplementedError) as raised:
                rebound_compiled(torch.tensor([-1.0, 2.0]))
            self.assertEqual(str(raised.exception), UNSUPPORTED_MESSAGE)

            with self.assertRaises(NotImplementedError) as raised:
                compiled(torch.tensor([-1.0, 2.0]))
            self.assertEqual(str(raised.exception), UNSUPPORTED_MESSAGE)
        finally:
            torch.relu = original_relu

        self.assertEqual(compiled(torch.tensor([-1.0, 2.0])).tolist(), [0.0, 2.0])

    def test_unary_relu_graphlet_keeps_callable_backends_unsupported(self):
        backend_calls = []

        def backend(graph_module, example_inputs):
            backend_calls.append((graph_module, example_inputs))
            return graph_module.forward

        def model(x):
            return x.relu()

        compiled = torch.compile(model, backend=backend)
        self.assertIs(compiled._torch_rs_compile_backend, backend)
        self.assertFalse(hasattr(compiled, "_torch_rs_compile_graph"))
        with self.assertRaises(NotImplementedError) as raised:
            compiled(torch.tensor([-1.0, 2.0]))
        self.assertEqual(str(raised.exception), UNSUPPORTED_MESSAGE)
        self.assertEqual(backend_calls, [])

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
