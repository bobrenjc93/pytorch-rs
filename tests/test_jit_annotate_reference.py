import copy
import importlib
import inspect
import pickle
import pickletools
import types
import typing
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class JitAnnotateReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "jit.annotate differentials require pinned PyTorch 2.13.0"
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

    def tensor_outcome(self, module):
        leaf = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        view = (leaf * 3.0).transpose(0, 1)[1]
        before = (
            tuple(view.shape),
            view.stride(),
            view.storage_offset(),
            view.data_ptr(),
            view.requires_grad,
            view.is_leaf,
        )
        result = module.jit.annotate(str, view)
        after = (
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
            result.data_ptr(),
            result.requires_grad,
            result.is_leaf,
        )
        result.sum().backward()
        return result is view, before == after, leaf.grad.tolist()

    def value_outcome(self, module):
        values = ([], {}, ({"nested": []},), object(), None)
        type_hints = (dict[str, int], module.Tensor, int, "wrong", object())
        return tuple(
            module.jit.annotate(type_hint, value) is value
            for type_hint, value in zip(type_hints, values, strict=True)
        )

    def test_tensor_view_autograd_container_and_mismatch_semantics_match(self):
        self.assertEqual(
            self.tensor_outcome(torch), self.tensor_outcome(reference_torch)
        )
        self.assertEqual(self.value_outcome(torch), self.value_outcome(reference_torch))

    def test_signature_documentation_and_identity_match(self):
        actual_jit = importlib.import_module("torch_rs.jit")
        expected_jit = importlib.import_module("torch.jit")
        actual = actual_jit.annotate
        expected = expected_jit.annotate

        self.assertIs(torch.jit, actual_jit)
        self.assertIs(reference_torch.jit, expected_jit)
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
        self.assertIs(inspect.getmodule(actual), actual_jit)
        self.assertIs(inspect.getmodule(expected), expected_jit)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )
        self.assertEqual(actual_jit.__doc__, expected_jit.__doc__)

    def test_exports_copy_and_pickle_match_the_supported_scope(self):
        actual_jit = torch.jit
        expected_jit = reference_torch.jit
        actual = actual_jit.annotate
        expected = expected_jit.annotate

        self.assertEqual(
            actual_jit.__all__,
            [
                name
                for name in expected_jit.__all__
                if name
                in {
                    "Attribute",
                    "annotate",
                    "export",
                    "ignore",
                    "isinstance",
                    "script_if_tracing",
                    "unused",
                }
            ],
        )
        self.assertEqual(
            torch.__all__.count("jit"),
            reference_torch.__all__.count("jit"),
        )
        self.assertEqual(
            torch.__all__.count("annotate"),
            reference_torch.__all__.count("annotate"),
        )

        actual_namespace = {}
        expected_namespace = {}
        exec("from torch_rs.jit import *", actual_namespace)
        exec("from torch.jit import *", expected_namespace)
        self.assertEqual(
            {name for name in actual_namespace if not name.startswith("__")},
            {
                "Attribute",
                "annotate",
                "export",
                "ignore",
                "isinstance",
                "script_if_tracing",
                "unused",
            },
        )
        self.assertIs(actual_namespace["annotate"], actual)
        self.assertIs(expected_namespace["annotate"], expected)

        self.assertIs(copy.copy(actual), actual)
        self.assertIs(copy.copy(expected), expected)
        self.assertIs(copy.deepcopy(actual), actual)
        self.assertIs(copy.deepcopy(expected), expected)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )
                self.assertIs(
                    pickle.loads(pickle.dumps(actual, protocol=protocol)), actual
                )
                self.assertIs(
                    pickle.loads(pickle.dumps(expected, protocol=protocol)), expected
                )

    def test_call_errors_match_pytorch_2_13(self):
        actual = torch.jit.annotate
        expected = reference_torch.jit.annotate
        cases = (
            (lambda function: function()),
            (lambda function: function(int)),
            (lambda function: function(the_value=1)),
            (lambda function: function(int, 1, 2)),
            (lambda function: function(type=int, the_value=1)),
            (lambda function: function(int, 1, the_type=str)),
            (lambda function: function(int, 1, the_value=2)),
        )
        for call in cases:
            with self.subTest(call=call):
                self.assert_error_matches(
                    lambda: call(actual),
                    lambda: call(expected),
                )

    def test_supported_boundary_is_eager_jit_helpers_only(self):
        expected_public = {
            name for name in vars(reference_torch.jit) if not name.startswith("_")
        }
        self.assertEqual(
            {name for name in vars(torch.jit) if not name.startswith("_")},
            {
                "Attribute",
                "Final",
                "annotate",
                "export",
                "ignore",
                "isinstance",
                "is_scripting",
                "is_tracing",
                "script_if_tracing",
                "unused",
            },
        )
        for name in ("script", "trace"):
            with self.subTest(name=name):
                self.assertIn(name, expected_public)
                self.assertFalse(hasattr(torch.jit, name))

        self.assertIs(torch.jit.is_scripting(), False)

        self.assertTrue(hasattr(reference_torch, "compile"))
        self.assertFalse(hasattr(torch, "compile"))


if __name__ == "__main__":
    unittest.main()
