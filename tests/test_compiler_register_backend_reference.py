import copy
import os
import pickle
import unittest

import torch_rs as torch
from torch_rs import _compiler_state as _state

if __package__:
    from .signature_utils import expose_reference_compiler_register_backend
else:
    from signature_utils import expose_reference_compiler_register_backend

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None

expose_reference_compiler_register_backend(reference_torch)


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CompilerRegisterBackendReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "compiler.register_backend differentials require pinned "
                "PyTorch 2.13.0"
            )

    def setUp(self):
        expose_reference_compiler_register_backend(reference_torch)
        self._registered_backends = dict(_state.registered_backends)
        self._registered_backend_fns = dict(_state.registered_backend_fns)
        _state.registered_backends.clear()
        _state.registered_backend_fns.clear()
        self.prefix = f"zz_register_{os.getpid()}_{self._testMethodName}"

    def tearDown(self):
        _state.registered_backends.clear()
        _state.registered_backends.update(self._registered_backends)
        _state.registered_backend_fns.clear()
        _state.registered_backend_fns.update(self._registered_backend_fns)

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def filtered_backends(self, module, exclude_tags):
        return [
            name
            for name in module.compiler.list_backends(exclude_tags)
            if name.startswith(self.prefix)
        ]

    def register_tagged_backends(self, module):
        def visible_backend(graph_module, example_inputs):
            return graph_module.forward

        def debug_backend(graph_module, example_inputs):
            return graph_module.forward

        def experimental_backend(graph_module, example_inputs):
            return graph_module.forward

        module.compiler.register_backend(
            visible_backend,
            name=f"{self.prefix}_visible",
        )
        module.compiler.register_backend(
            debug_backend,
            name=f"{self.prefix}_debug",
            tags=("debug",),
        )
        module.compiler.register_backend(
            experimental_backend,
            name=f"{self.prefix}_experimental",
            tags=("experimental",),
        )

    def test_direct_decorator_duplicate_and_non_callable_forms_match_pytorch_2_13(self):
        for module in (torch, reference_torch):
            with self.subTest(module=module.__name__):
                calls = []

                def direct_backend(graph_module, example_inputs):
                    calls.append(("direct", graph_module, example_inputs))
                    return graph_module.forward

                self.assertIs(
                    module.compiler.register_backend(
                        direct_backend,
                        name=f"{self.prefix}_{module.__name__}_direct",
                        tags=("debug",),
                    ),
                    direct_backend,
                )

                @module.compiler.register_backend(
                    name=f"{self.prefix}_{module.__name__}_decorated",
                    tags=["experimental"],
                )
                def decorated_backend(graph_module, example_inputs):
                    calls.append(("decorated", graph_module, example_inputs))
                    return graph_module.forward

                self.assertIsNot(decorated_backend._tags, None)
                self.assertEqual(direct_backend._tags, ("debug",))
                self.assertEqual(decorated_backend._tags, ("experimental",))
                self.assertEqual(calls, [])

        def actual_base_backend(graph_module, example_inputs):
            return graph_module.forward

        def reference_base_backend(graph_module, example_inputs):
            return graph_module.forward

        def actual_duplicate_backend(graph_module, example_inputs):
            return graph_module.forward

        def reference_duplicate_backend(graph_module, example_inputs):
            return graph_module.forward

        duplicate_name = f"{self.prefix}_duplicate"
        torch.compiler.register_backend(actual_base_backend, name=duplicate_name)
        reference_torch.compiler.register_backend(
            reference_base_backend,
            name=duplicate_name,
        )
        self.assert_error_matches(
            lambda: torch.compiler.register_backend(
                actual_duplicate_backend,
                name=duplicate_name,
            ),
            lambda: reference_torch.compiler.register_backend(
                reference_duplicate_backend,
                name=duplicate_name,
            ),
        )
        self.assert_error_matches(
            lambda: torch.compiler.register_backend(
                42,
                name=f"{self.prefix}_int",
            ),
            lambda: reference_torch.compiler.register_backend(
                42,
                name=f"{self.prefix}_int",
            ),
        )

    def test_tag_filtering_matches_pytorch_2_13_for_registered_names(self):
        self.register_tagged_backends(torch)
        self.register_tagged_backends(reference_torch)

        cases = (
            (),
            None,
            ("debug",),
            ("experimental",),
            ("debug", "experimental"),
            "debug",
        )
        for exclude_tags in cases:
            with self.subTest(exclude_tags=exclude_tags):
                self.assertEqual(
                    self.filtered_backends(torch, exclude_tags),
                    self.filtered_backends(reference_torch, exclude_tags),
                )

    def test_copy_pickle_and_unsupported_execution_boundaries_match_supported_scope(self):
        actual = torch.compiler.register_backend
        expected = reference_torch.compiler.register_backend

        for function in (actual, expected):
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol)),
                    function,
                )

        self.assertTrue(callable(torch.compile))
        self.assertFalse(hasattr(torch, "export"))
        self.assertFalse(hasattr(torch, "register_backend"))
        self.assertFalse(hasattr(torch.compiler, "compile"))
        self.assertTrue(callable(torch.compiler.register_backend))
        self.assertTrue(callable(reference_torch.compile))


if __name__ == "__main__":
    unittest.main()
