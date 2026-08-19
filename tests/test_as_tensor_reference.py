import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class AsTensorReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("as_tensor differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(
            type(actual_raised.exception).__name__,
            type(expected_raised.exception).__name__,
        )
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def identity_contract(self, module):
        base = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
            requires_grad=True,
        )
        sources = (
            base,
            base[1, 2, 3],
            base.transpose(0, 2),
            base.transpose(0, 2)[1],
            module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
        )
        calls = (
            lambda source: module.as_tensor(source),
            lambda source: module.as_tensor(data=source),
            lambda source: module.as_tensor(source, dtype=None),
            lambda source: module.as_tensor(source, dtype=module.float32),
            lambda source: module.as_tensor(source, device=None),
            lambda source: module.as_tensor(source, device="cpu"),
            lambda source: module.as_tensor(source, device=module.device("cpu")),
            lambda source: module.as_tensor(
                data=source, dtype=module.float32, device="cpu"
            ),
        )
        observations = []
        for source in sources:
            for call in calls:
                result = call(source)
                observations.append(
                    (
                        result is source,
                        tuple(result.shape),
                        result.stride(),
                        result.storage_offset(),
                        result.data_ptr() == source.data_ptr(),
                        str(result.dtype),
                        str(result.device),
                        str(result.layout),
                        result.requires_grad,
                        result.is_leaf,
                        result.output_nr,
                    )
                )
        return tuple(observations)

    def test_identity_layout_storage_and_autograd_metadata_match_pytorch_2_13(self):
        self.assertEqual(
            self.identity_contract(torch),
            self.identity_contract(reference_torch),
        )

        def autograd_contract(module):
            leaf = module.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                dtype=module.float32,
                requires_grad=True,
            )
            view = leaf.transpose(0, 1)[1]
            result = module.as_tensor(
                view, dtype=module.float32, device=module.device("cpu")
            )
            (result * module.tensor([3.0, 7.0], dtype=module.float32)).sum().backward()
            return (
                result is view,
                result.data_ptr() == view.data_ptr(),
                result.storage_offset(),
                result.stride(),
                result.requires_grad,
                result.is_leaf,
                leaf.grad.tolist(),
            )

        self.assertEqual(
            autograd_contract(torch), autograd_contract(reference_torch)
        )

    def test_binding_validation_and_error_precedence_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor(
            [1.0], dtype=reference_torch.float32
        )
        call_pairs = (
            (lambda: torch.as_tensor(), lambda: reference_torch.as_tensor()),
            (
                lambda: torch.as_tensor(dtype=1),
                lambda: reference_torch.as_tensor(dtype=1),
            ),
            (
                lambda: torch.as_tensor(actual, None),
                lambda: reference_torch.as_tensor(expected, None),
            ),
            (
                lambda: torch.as_tensor(actual, data=actual),
                lambda: reference_torch.as_tensor(expected, data=expected),
            ),
            (
                lambda: torch.as_tensor(actual, unexpected=True),
                lambda: reference_torch.as_tensor(expected, unexpected=True),
            ),
            (
                lambda: torch.as_tensor(actual, dtype=1),
                lambda: reference_torch.as_tensor(expected, dtype=1),
            ),
            (
                lambda: torch.as_tensor(actual, device=1.5),
                lambda: reference_torch.as_tensor(expected, device=1.5),
            ),
            (
                lambda: torch.as_tensor(actual, dtype=1, unexpected=True),
                lambda: reference_torch.as_tensor(
                    expected, dtype=1, unexpected=True
                ),
            ),
            (
                lambda: torch.as_tensor(actual, device=1.5, unexpected=True),
                lambda: reference_torch.as_tensor(
                    expected, device=1.5, unexpected=True
                ),
            ),
            (
                lambda: torch.as_tensor(actual, data=actual, dtype=1),
                lambda: reference_torch.as_tensor(
                    expected, data=expected, dtype=1
                ),
            ),
            (
                lambda: torch.as_tensor(actual, device=""),
                lambda: reference_torch.as_tensor(expected, device=""),
            ),
            (
                lambda: torch.as_tensor(actual, device="banana"),
                lambda: reference_torch.as_tensor(expected, device="banana"),
            ),
            (
                lambda: torch.as_tensor(actual, device="cpu:01"),
                lambda: reference_torch.as_tensor(expected, device="cpu:01"),
            ),
            (
                lambda: torch.as_tensor(actual, device="cpu:2147483648"),
                lambda: reference_torch.as_tensor(
                    expected, device="cpu:2147483648"
                ),
            ),
        )
        for actual_call, expected_call in call_pairs:
            with self.subTest(actual_call=actual_call):
                self.assert_error_matches(actual_call, expected_call)

    def mode_contract(self, module):
        function = module.as_tensor
        source = module.tensor([1.0], dtype=module.float32)
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        def normalize_call(call):
            func, dispatch_types, args, kwargs = call
            if kwargs is None:
                normalized_kwargs = None
            else:
                def normalize_value(value):
                    if value is source:
                        return "source"
                    if value is module.float32:
                        return "float32"
                    if isinstance(value, str):
                        return ("str", value)
                    if isinstance(value, list):
                        return ("list", tuple(value))
                    return type(value).__name__

                normalized_kwargs = tuple(
                    (key, normalize_value(value))
                    for key, value in kwargs.items()
                )
            return (
                func is function,
                tuple(item.__name__ for item in dispatch_types),
                len(args),
                tuple(item is source for item in args),
                normalized_kwargs,
            )

        accepting = RecordingMode(marker)
        with accepting:
            accepted = function(source)

        malformed = RecordingMode(marker)
        with malformed:
            malformed_result = function(data=[1.0], device="not-a-device")

        invalid = RecordingMode(marker)
        invalid_errors = []
        for call in (
            lambda: function(source, dtype=1),
            lambda: function(source, unexpected=True),
        ):
            try:
                with invalid:
                    call()
            except Exception as error:
                invalid_errors.append((type(error).__name__, str(error)))

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append((self.label, func, types, args, kwargs))
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = function(
                    source, dtype=module.float32, device="cpu"
                )

        declining = RecordingMode(NotImplemented)
        try:
            with declining:
                function(source)
        except Exception as error:
            declining_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x<address>", str(error)),
            )
        else:
            self.fail(f"{module.__name__} accepted a declining mode")

        return {
            "accepting": (
                accepted is marker,
                tuple(map(normalize_call, accepting.calls)),
            ),
            "malformed": (
                malformed_result is marker,
                tuple(map(normalize_call, malformed.calls)),
            ),
            "invalid_errors": tuple(invalid_errors),
            "invalid_calls": tuple(map(normalize_call, invalid.calls)),
            "order": tuple(
                (label, normalize_call((func, types, args, kwargs)))
                for label, func, types, args, kwargs in order
            ),
            "forwarded": forwarded is source,
            "declining": declining_error,
            "declining_calls": tuple(map(normalize_call, declining.calls)),
            "stack_depth": len(module.overrides._get_current_function_mode_stack()),
        }

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_contract(torch),
            self.mode_contract(reference_torch),
        )

    def callable_contract(self, module):
        function = module.as_tensor
        reducer, (owner, name) = function.__reduce__()
        namespace = {}
        exec(f"from {module.__name__} import *", namespace)
        try:
            inspect.signature(function)
        except Exception as error:
            signature_error = (
                type(error).__name__,
                str(error).split(" for ", 1)[0],
            )
        else:
            signature_error = None
        return {
            "type": type(function).__name__,
            "is_builtin": type(function) is types.BuiltinFunctionType,
            "name": function.__name__,
            "qualname": function.__qualname__,
            "module": function.__module__,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "self_is_none": function.__self__ is None,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "reducer_is_getattr": reducer is getattr,
            "reduce_name": name,
            "owner_name": owner.__name__,
            "owner_qualname": owner.__qualname__,
            "owner_module": owner.__module__.replace("torch_rs._C", "torch._C"),
            "owner_identity": owner is module._C._VariableFunctionsClass,
            "owner_function_identity": owner.as_tensor is function,
            "all_count": module.__all__.count("as_tensor"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": namespace["as_tensor"] is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_documentation_exports_and_pickling_match(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )

    def test_conversion_boundaries_are_explicit(self):
        actual = torch.tensor([1.0, 2.0])
        expected = reference_torch.tensor(
            [1.0, 2.0], dtype=reference_torch.float32, requires_grad=True
        )

        with self.assertRaisesRegex(RuntimeError, "indexed CPU device"):
            torch.as_tensor(actual, device="cpu:0")
        indexed = reference_torch.as_tensor(expected, device="cpu:0")
        self.assertIsNot(indexed, expected)
        self.assertNotEqual(indexed.data_ptr(), expected.data_ptr())
        self.assertEqual(indexed.tolist(), expected.tolist())
        indexed.sum().backward()
        self.assertEqual(expected.grad.tolist(), [1.0, 1.0])

        with self.assertRaises(TypeError):
            torch.as_tensor(actual, dtype=reference_torch.float64)
        converted = reference_torch.as_tensor(
            expected.detach(), dtype=reference_torch.float64
        )
        self.assertIsNot(converted, expected)
        self.assertIs(converted.dtype, reference_torch.float64)

    def test_visible_accelerator_target_is_rejected_at_the_native_boundary(self):
        if not reference_torch.cuda.is_available():
            self.skipTest("requires a visible CUDA accelerator")

        actual = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).transpose(0, 1)
        with self.assertRaisesRegex(RuntimeError, "device 'cuda' is not supported"):
            torch.as_tensor(actual, device="cuda")

        leaf = reference_torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        source = leaf.transpose(0, 1)
        converted = reference_torch.as_tensor(source, device="cuda")
        self.assertTrue(converted.is_cuda)
        self.assertEqual(converted.device.index, 0)
        self.assertIsNot(converted, source)
        self.assertEqual(converted.stride(), source.stride())
        self.assertEqual(converted.cpu().tolist(), source.tolist())
        converted.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[1.0, 1.0], [1.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
