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
from torch_rs import _compiler_state as _state


UNSUPPORTED_MESSAGE = (
    "torch.compile(): graph capture, graph execution, and eager fallback are "
    "not supported; only argument binding, disable=True pass-through, and "
    "backend resolution are implemented"
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

    def test_identity_function_compiles_backend_once_and_runs_backend_callable(self):
        backend_calls = []
        compiled_calls = []

        def model(value):
            return value

        def backend(graph_module, example_inputs):
            backend_calls.append((graph_module, example_inputs))

            def compiled(value):
                compiled_calls.append(value)
                return graph_module.forward(value)

            return compiled

        first = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        second = torch.tensor([5.0, 6.0], dtype=torch.float32)
        compiled = torch.compile(model, backend=backend, name="identity_graph")

        self.assertIs(compiled(first), first)

        self.assertEqual(len(backend_calls), 1)
        graph_module, example_inputs = backend_calls[0]
        self.assertEqual(len(example_inputs), 1)
        proxy = example_inputs[0]
        self.assertIsNot(proxy, first)
        self.assertEqual(type(proxy).__module__, "torch_rs")
        self.assertEqual(type(proxy).__name__, "_CompileTensorProxy")
        self.assertIs(proxy.dtype, torch.float32)
        self.assertEqual(proxy.device, torch.device("cpu"))
        self.assertEqual(proxy.shape, (2, 2))
        self.assertEqual(type(graph_module).__module__, "torch_rs")
        self.assertEqual(type(graph_module).__name__, "_CompileIdentityGraph")
        self.assertEqual(graph_module.name, "identity_graph")
        self.assertEqual(graph_module.inputs, ("arg0",))
        self.assertEqual(graph_module.outputs, ("arg0",))
        self.assertEqual(graph_module.operation, "identity")
        self.assertIs(graph_module.forward(first), first)
        self.assertIs(example_inputs[0], proxy)

        self.assertEqual(len(compiled_calls), 1)
        self.assertIs(compiled_calls[0], first)

        self.assertIs(compiled(second), second)
        self.assertEqual(len(backend_calls), 1)
        self.assertEqual(len(compiled_calls), 2)
        self.assertIs(compiled_calls[1], second)

        torch.compiler.register_backend(backend, name="zz_identity")
        named_compiled = torch.compile(model, backend="zz_identity")
        third = torch.tensor([7.0], dtype=torch.float32)
        self.assertIs(named_compiled(third), third)
        self.assertEqual(len(backend_calls), 2)
        self.assertEqual(len(compiled_calls), 3)
        self.assertIs(compiled_calls[2], third)

    def test_identity_decorator_form_compiles_and_non_identity_stays_unsupported(
        self,
    ):
        backend_calls = []

        def backend(graph_module, example_inputs):
            backend_calls.append((graph_module, example_inputs))
            return graph_module.forward

        @torch.compile(backend=backend)
        def decorated_identity(value):
            return value

        input_tensor = torch.tensor([1.25, -2.5], dtype=torch.float32)
        self.assertIs(decorated_identity(input_tensor), input_tensor)
        self.assertEqual(len(backend_calls), 1)

        def non_identity(value):
            return value.neg()

        compiled_non_identity = torch.compile(non_identity, backend=backend)
        with self.assertRaises(NotImplementedError) as raised:
            compiled_non_identity(input_tensor)
        self.assertEqual(str(raised.exception), UNSUPPORTED_MESSAGE)
        self.assertEqual(len(backend_calls), 1)

        def metadata_conditional(value):
            if value.size(0) == 2:
                return value
            return value.neg()

        compiled_conditional = torch.compile(metadata_conditional, backend=backend)
        length_two = torch.tensor([1.0, 2.0], dtype=torch.float32)
        with self.assertRaises(NotImplementedError) as raised:
            compiled_conditional(length_two)
        self.assertEqual(str(raised.exception), UNSUPPORTED_MESSAGE)
        self.assertEqual(len(backend_calls), 1)

        def proxy_type_conditional(value):
            if not isinstance(value, torch.Tensor):
                return value
            return value.neg()

        compiled_proxy_type = torch.compile(proxy_type_conditional, backend=backend)
        with self.assertRaises(NotImplementedError) as raised:
            compiled_proxy_type(input_tensor)
        self.assertEqual(str(raised.exception), UNSUPPORTED_MESSAGE)
        self.assertEqual(len(backend_calls), 1)

    def test_identity_backend_is_invoked_once_under_concurrent_first_calls(self):
        backend_entered = threading.Event()
        release_backend = threading.Event()
        second_worker_started = threading.Event()
        backend_calls = []
        results = []
        errors = []

        def model(value):
            return value

        def backend(graph_module, example_inputs):
            backend_calls.append((graph_module, example_inputs))
            backend_entered.set()
            if not release_backend.wait(timeout=10):
                raise RuntimeError("timed out waiting to release backend")
            return graph_module.forward

        compiled = torch.compile(model, backend=backend)
        inputs = (
            torch.tensor([1.0], dtype=torch.float32),
            torch.tensor([2.0], dtype=torch.float32),
        )

        def worker(input_tensor, started=None):
            try:
                if started is not None:
                    started.set()
                results.append(compiled(input_tensor))
            except BaseException as error:
                errors.append(error)

        first_thread = threading.Thread(target=worker, args=(inputs[0],))
        second_thread = threading.Thread(
            target=worker,
            args=(inputs[1], second_worker_started),
        )
        first_thread.start()
        self.assertTrue(backend_entered.wait(timeout=10))
        second_thread.start()
        self.assertTrue(second_worker_started.wait(timeout=10))
        release_backend.set()

        threads = [first_thread, second_thread]
        for thread in threads:
            thread.join(timeout=10)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(set(map(id, results)), {id(input) for input in inputs})
        self.assertEqual(len(backend_calls), 1)

    def test_compiler_reset_clears_identity_backend_cache(self):
        backend_calls = []

        def model(value):
            return value

        def backend(graph_module, example_inputs):
            backend_calls.append((graph_module, example_inputs))
            return graph_module.forward

        compiled = torch.compile(model, backend=backend)
        input_tensor = torch.tensor([1.0], dtype=torch.float32)
        self.assertIs(compiled(input_tensor), input_tensor)
        self.assertEqual(len(backend_calls), 1)

        self.assertIs(torch.compiler.reset(), None)
        self.assertIs(compiled(input_tensor), input_tensor)
        self.assertEqual(len(backend_calls), 2)

    def test_compiled_identity_wrapper_survives_reload_before_first_call(self):
        backend_calls = []

        def model(value):
            return value

        def backend(graph_module, example_inputs):
            backend_calls.append((graph_module, example_inputs))
            return graph_module.forward

        compiled = torch.compile(model, backend=backend)
        self.assertIs(importlib.reload(torch), torch)

        input_tensor = torch.tensor([1.0], dtype=torch.float32)
        self.assertIs(compiled(input_tensor), input_tensor)
        self.assertEqual(len(backend_calls), 1)

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

def identity(value):
    return value

def identity_backend(graph_module, example_inputs):
    calls.append((
        "identity_backend",
        type(graph_module).__module__,
        type(graph_module).__name__,
        graph_module.operation,
        len(example_inputs),
        type(example_inputs[0]).__module__,
        type(example_inputs[0]).__name__,
    ))
    return graph_module.forward

tensor = torch.tensor([1.0], dtype=torch.float32)
compiled_identity = torch.compile(identity, backend=identity_backend)
assert compiled_identity(tensor) is tensor
assert calls == [
    (
        "identity_backend",
        "torch_rs",
        "_CompileIdentityGraph",
        "identity",
        1,
        "torch_rs",
        "_CompileTensorProxy",
    ),
]
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
