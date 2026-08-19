import copy
import importlib
import inspect
import pickle
import re
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class AutogradBackwardTests(unittest.TestCase):
    def test_canonical_imports_exports_copy_and_pickle(self):
        autograd = importlib.import_module("torch_rs.autograd")
        from torch_rs.autograd import backward

        function = autograd.backward
        self.assertIs(torch.autograd, autograd)
        self.assertIs(backward, function)
        self.assertFalse(hasattr(torch, "backward"))
        self.assertEqual(autograd.__all__, ["backward", "grad_mode", "no_grad"])

        wildcard_namespace = {}
        exec("from torch_rs.autograd import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["backward"], function)

        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "backward")
        self.assertEqual(function.__qualname__, "backward")
        self.assertEqual(function.__module__, "torch_rs.autograd")
        self.assertIs(inspect.getmodule(function), autograd)
        self.assertEqual(function.__defaults__, (None, None, False, None, None))
        self.assertIsNone(function.__kwdefaults__)
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.autograd", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_direct_tensor_and_explicit_default_forms_return_none(self):
        calls = (
            lambda output: torch.autograd.backward(output),
            lambda output: torch.autograd.backward(
                output, None, None, False, None, None
            ),
            lambda output: torch.autograd.backward(
                tensors=output,
                grad_tensors=None,
                retain_graph=None,
                create_graph=False,
                grad_variables=None,
                inputs=None,
            ),
            lambda output: torch.autograd.backward(output, retain_graph=False),
            lambda output: torch.autograd.backward(output, retain_graph=0),
            lambda output: torch.autograd.backward(output, create_graph=0),
        )
        for index, call in enumerate(calls):
            with self.subTest(index=index):
                leaf = torch.tensor(3.0, requires_grad=True)
                result = call(leaf * leaf)
                self.assertIsNone(result)
                self.assertEqual(leaf.grad.item(), 6.0)

    def test_gradient_accumulation_and_freed_graph_error(self):
        leaf = torch.tensor(2.0, requires_grad=True)
        first = leaf * leaf
        self.assertIsNone(torch.autograd.backward(first))
        retained_gradient = leaf.grad
        self.assertEqual(retained_gradient.item(), 4.0)

        self.assertIsNone(torch.autograd.backward(leaf * 3.0))
        self.assertIs(leaf.grad, retained_gradient)
        self.assertEqual(leaf.grad.item(), 7.0)

        with self.assertRaisesRegex(
            RuntimeError,
            "^Trying to backward through the graph a second time",
        ):
            torch.autograd.backward(first)

    def test_advanced_forms_remain_explicitly_unsupported(self):
        tensor = torch.tensor(2.0, requires_grad=True)
        cases = (
            (
                lambda: torch.autograd.backward([tensor]),
                "torch_rs.autograd.backward only supports a single Tensor",
            ),
            (
                lambda: torch.autograd.backward((tensor,)),
                "torch_rs.autograd.backward only supports a single Tensor",
            ),
            (
                lambda: torch.autograd.backward(
                    tensor, grad_tensors=torch.tensor(1.0)
                ),
                "torch_rs.autograd.backward does not support explicit gradients",
            ),
            (
                lambda: torch.autograd.backward(
                    tensor, grad_variables=torch.tensor(1.0)
                ),
                "torch_rs.autograd.backward does not support explicit gradients",
            ),
            (
                lambda: torch.autograd.backward(tensor, retain_graph=True),
                "torch_rs.autograd.backward does not support retained graphs",
            ),
            (
                lambda: torch.autograd.backward(tensor, create_graph=True),
                "torch_rs.autograd.backward does not support higher-order graphs",
            ),
            (
                lambda: torch.autograd.backward(tensor, inputs=tensor),
                "torch_rs.autograd.backward does not support input filtering",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(NotImplementedError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class AutogradBackwardReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "autograd.backward differentials require pinned PyTorch 2.13.0"
            )

    def supported_contract(self, module):
        results = []
        calls = (
            lambda output: module.autograd.backward(output),
            lambda output: module.autograd.backward(
                output, None, None, False, None, None
            ),
            lambda output: module.autograd.backward(
                tensors=output,
                grad_tensors=None,
                retain_graph=None,
                create_graph=False,
                grad_variables=None,
                inputs=None,
            ),
            lambda output: module.autograd.backward(output, retain_graph=False),
            lambda output: module.autograd.backward(output, retain_graph=0),
            lambda output: module.autograd.backward(output, create_graph=0),
        )
        for call in calls:
            leaf = module.tensor(3.0, requires_grad=True)
            result = call(leaf * leaf)
            results.append((result, leaf.grad.item()))

        leaf = module.tensor(2.0, requires_grad=True)
        consumed = leaf * leaf
        module.autograd.backward(consumed)
        first_gradient = leaf.grad
        module.autograd.backward(leaf * 3.0)
        accumulation = (
            first_gradient.item(),
            leaf.grad.item(),
            first_gradient is leaf.grad,
        )
        try:
            module.autograd.backward(consumed)
        except Exception as error:
            freed_graph = (type(error).__name__, str(error))
        else:
            freed_graph = None
        return tuple(results), accumulation, freed_graph

    def test_supported_semantics_match_pytorch_2_13(self):
        self.assertEqual(
            self.supported_contract(torch),
            self.supported_contract(reference_torch),
        )

    def scalar_boundary_contract(self, module):
        outcomes = []
        for shape in ((2,), (0,), (1,), ()):
            tensor = module.ones(shape, requires_grad=True)
            try:
                result = module.autograd.backward(tensor)
            except Exception as error:
                outcomes.append((shape, type(error).__name__, str(error)))
            else:
                outcomes.append((shape, result, tensor.grad.tolist()))
        return tuple(outcomes)

    def test_scalar_and_non_scalar_boundaries_match_pytorch_2_13(self):
        self.assertEqual(
            self.scalar_boundary_contract(torch),
            self.scalar_boundary_contract(reference_torch),
        )

    def mode_contract(self, module):
        function = module.autograd.backward
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        def normalize_call(call, output):
            func, dispatch_types, args, kwargs = call
            return (
                func is function,
                tuple(item.__name__ for item in dispatch_types),
                len(args) == 1
                and type(args[0]) is tuple
                and len(args[0]) == 1
                and args[0][0] is output,
                kwargs,
            )

        accepting_leaf = module.tensor(2.0, requires_grad=True)
        accepting_output = accepting_leaf * accepting_leaf
        accepting = RecordingMode(marker)
        with accepting:
            accepting_result = function(accepting_output)
        accepting_untouched = accepting_leaf.grad is None
        function(accepting_output)

        multi_element = module.ones((2,), requires_grad=True)
        multi_accepting = RecordingMode(marker)
        with multi_accepting:
            multi_result = function(multi_element)

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append((self.label, func, types, args, kwargs))
                return func(*args, **(kwargs or {}))

        forwarding_leaf = module.tensor(3.0, requires_grad=True)
        forwarding_output = forwarding_leaf * forwarding_leaf
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = function(forwarding_output)

        declining_leaf = module.tensor(4.0, requires_grad=True)
        declining_output = declining_leaf * declining_leaf
        declining = RecordingMode(NotImplemented)
        try:
            with declining:
                function(declining_output)
        except Exception as error:
            declining_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x<address>", str(error)).replace(
                    "torch_rs", "torch"
                ),
            )
        else:
            self.fail(f"{module.__name__} accepted a declining mode")
        declining_untouched = declining_leaf.grad is None

        return {
            "accepting": (
                accepting_result is marker,
                accepting_untouched,
                accepting_leaf.grad.item(),
                tuple(
                    normalize_call(call, accepting_output)
                    for call in accepting.calls
                ),
            ),
            "multi_element": (
                multi_result is marker,
                multi_element.grad is None,
                tuple(
                    normalize_call(call, multi_element)
                    for call in multi_accepting.calls
                ),
            ),
            "forwarding": tuple(
                (
                    label,
                    normalize_call(
                        (func, dispatch_types, args, kwargs), forwarding_output
                    ),
                )
                for label, func, dispatch_types, args, kwargs in order
            ),
            "forwarded": (forwarded, forwarding_leaf.grad.item()),
            "declining": (
                declining_error,
                declining_untouched,
                tuple(
                    normalize_call(call, declining_output)
                    for call in declining.calls
                ),
            ),
            "stack_depth": len(module.overrides._get_current_function_mode_stack()),
        }

    def test_torch_function_modes_match_pytorch_2_13(self):
        self.assertEqual(
            self.mode_contract(torch),
            self.mode_contract(reference_torch),
        )

    def test_signature_annotations_metadata_docs_exports_and_pickle_match(self):
        actual_module = torch.autograd
        expected_module = reference_torch.autograd
        actual = actual_module.backward
        expected = expected_module.backward

        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertIs(inspect.getmodule(actual), actual_module)
        self.assertIs(inspect.getmodule(expected), expected_module)
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(
            str(actual.__annotations__).replace("torch_rs", "torch"),
            str(expected.__annotations__),
        )
        self.assertEqual(
            str(inspect.signature(actual)).replace("torch_rs", "torch"),
            str(inspect.signature(expected)),
        )
        self.assertEqual(
            actual_module.__all__,
            [
                name
                for name in expected_module.__all__
                if name in {"backward", "grad_mode", "no_grad"}
            ],
        )

        actual_wildcard = {}
        expected_wildcard = {}
        exec("from torch_rs.autograd import *", actual_wildcard)
        exec("from torch.autograd import *", expected_wildcard)
        self.assertIs(actual_wildcard["backward"], actual)
        self.assertIs(expected_wildcard["backward"], expected)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(actual, protocol=protocol)), actual
                )
                self.assertIs(
                    pickle.loads(pickle.dumps(expected, protocol=protocol)), expected
                )


if __name__ == "__main__":
    unittest.main()
