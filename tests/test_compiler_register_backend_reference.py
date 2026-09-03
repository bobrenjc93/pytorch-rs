import functools
import unittest

import torch_rs as torch
from torch_rs import _compiler_state

try:
    import torch as reference_torch
    import torch._dynamo.backends.registry as reference_registry
except ImportError:
    reference_torch = None
    reference_registry = None


@unittest.skipIf(reference_registry is None, "install the reference dependency group")
class CompilerRegisterBackendReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "compiler.register_backend differentials require pinned "
                "PyTorch 2.13.0"
            )

    def setUp(self):
        self._actual_registry = _compiler_state.registered_backends
        self._saved_actual_registry = dict(self._actual_registry)
        self._actual_registry.clear()
        self._reference_names = []

    def tearDown(self):
        self._actual_registry.clear()
        self._actual_registry.update(self._saved_actual_registry)
        for name in self._reference_names:
            reference_registry._BACKENDS.pop(name, None)
            reference_registry._COMPILER_FNS.pop(name, None)

    def registration_outcome(self, register_backend, list_backends, prefix):
        calls = []
        direct_name = f"{prefix}_direct"
        debug_name = f"{prefix}_debug"
        experimental_name = f"{prefix}_experimental"

        def direct_backend(graph_module, example_inputs):
            calls.append((graph_module, example_inputs))
            return graph_module.forward

        direct_result = register_backend(direct_backend, name=direct_name)

        decorator = register_backend(
            name=debug_name,
            tags=("debug", "custom"),
        )
        self.assertIs(type(decorator), functools.partial)

        @decorator
        def debug_backend(graph_module, example_inputs):
            calls.append((graph_module, example_inputs))
            return graph_module.forward

        def experimental_backend(graph_module, example_inputs):
            calls.append((graph_module, example_inputs))
            return graph_module.forward

        experimental_result = register_backend(
            experimental_backend,
            name=experimental_name,
            tags=("experimental",),
        )

        default_backends = list_backends()
        all_backends = list_backends(exclude_tags=())
        no_experimental_backends = list_backends(exclude_tags=("experimental",))
        no_custom_backends = list_backends(exclude_tags=("custom",))
        none_excluded_backends = list_backends(exclude_tags=None)

        return {
            "direct_result": direct_result is direct_backend,
            "debug_result": debug_backend.__name__,
            "experimental_result": experimental_result is experimental_backend,
            "debug_tags": debug_backend._tags,
            "experimental_tags": experimental_backend._tags,
            "direct_in_default": direct_name in default_backends,
            "debug_in_default": debug_name in default_backends,
            "experimental_in_default": experimental_name in default_backends,
            "all_visible": (
                direct_name in all_backends,
                debug_name in all_backends,
                experimental_name in all_backends,
            ),
            "exclude_experimental": (
                direct_name in no_experimental_backends,
                debug_name in no_experimental_backends,
                experimental_name in no_experimental_backends,
            ),
            "exclude_custom": (
                direct_name in no_custom_backends,
                debug_name in no_custom_backends,
                experimental_name in no_custom_backends,
            ),
            "none_excluded": (
                direct_name in none_excluded_backends,
                debug_name in none_excluded_backends,
                experimental_name in none_excluded_backends,
            ),
            "calls": calls,
        }

    def test_direct_decorator_and_tag_filtering_match_pytorch_2_13_registry(self):
        actual_prefix = "actual"
        expected_prefix = "expected"
        self._reference_names.extend(
            [
                f"{expected_prefix}_direct",
                f"{expected_prefix}_debug",
                f"{expected_prefix}_experimental",
            ]
        )

        self.assertEqual(
            self.registration_outcome(
                torch.compiler.register_backend,
                torch.compiler.list_backends,
                actual_prefix,
            ),
            self.registration_outcome(
                reference_registry.register_backend,
                reference_registry.list_backends,
                expected_prefix,
            ),
        )

    def error_outcome(self, register_backend, prefix):
        duplicate_name = f"{prefix}_duplicate"
        builtin_name = f"{prefix}_builtin"

        def backend(graph_module, example_inputs):
            return graph_module.forward

        def duplicate_backend(graph_module, example_inputs):
            return graph_module.forward

        register_backend(backend, name=duplicate_name)
        outcomes = []
        for call in (
            lambda: register_backend(duplicate_backend, name=duplicate_name),
            lambda: register_backend("not callable", name=f"{prefix}_bad"),
            lambda: register_backend(_CallableBackend()),
            lambda: register_backend(len, name=builtin_name),
        ):
            try:
                call()
            except BaseException as error:
                message = str(error).replace(prefix, "<prefix>")
                args = tuple(
                    arg.replace(prefix, "<prefix>") if isinstance(arg, str) else arg
                    for arg in error.args
                )
                outcomes.append((type(error).__name__, message, args))
            else:
                outcomes.append(None)
        return outcomes

    def test_duplicate_and_invalid_callables_match_pytorch_2_13_registry(self):
        actual_prefix = "actual_error"
        expected_prefix = "expected_error"
        self._reference_names.extend(
            [f"{expected_prefix}_duplicate", f"{expected_prefix}_builtin"]
        )

        self.assertEqual(
            self.error_outcome(torch.compiler.register_backend, actual_prefix),
            self.error_outcome(reference_registry.register_backend, expected_prefix),
        )


class _CallableBackend:
    def __call__(self, graph_module, example_inputs):
        return graph_module.forward


if __name__ == "__main__":
    unittest.main()
