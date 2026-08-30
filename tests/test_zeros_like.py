import inspect
import pickle
import re
import unittest

import torch_rs as torch


class ZerosLikeTests(unittest.TestCase):
    def tensor_contract(self, tensor):
        return {
            "shape": tuple(tensor.shape),
            "stride": tensor.stride(),
            "storage_offset": tensor.storage_offset(),
            "numel": tensor.numel(),
            "values": tensor.tolist(),
            "dtype": tensor.dtype,
            "device": tensor.device,
            "layout": tensor.layout,
            "requires_grad": tensor.requires_grad,
            "is_leaf": tensor.is_leaf,
            "grad_is_none": tensor.grad is None,
            "output_nr": tensor.output_nr,
        }

    def assert_zero_like(self, source, result, expected_values, *, requires_grad=False):
        self.assertIsNot(result, source)
        self.assertFalse(result.is_set_to(source))
        if source.numel() != 0:
            self.assertNotEqual(result.data_ptr(), source.data_ptr())
        self.assertEqual(
            self.tensor_contract(result),
            {
                "shape": tuple(source.shape),
                "stride": source.stride(),
                "storage_offset": 0,
                "numel": source.numel(),
                "values": expected_values,
                "dtype": source.dtype,
                "device": source.device,
                "layout": source.layout,
                "requires_grad": requires_grad,
                "is_leaf": True,
                "grad_is_none": True,
                "output_nr": 0,
            },
        )

    def test_scalar_empty_and_contiguous_metadata_values_and_fresh_storage(self):
        cases = (
            (torch.tensor(-3.5), 0.0),
            (torch.zeros((2, 0, 3)), [[], []]),
            (torch.ones((1, 3)).transpose(0, 1), [[0.0], [0.0], [0.0]]),
            (torch.zeros((2, 0, 3)).transpose(0, 2), [[], [], []]),
            (
                torch.tensor([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]]),
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ),
        )
        for source, expected_values in cases:
            with self.subTest(shape=tuple(source.shape)):
                self.assertTrue(source.is_contiguous())
                self.assert_zero_like(
                    source, torch.zeros_like(source), expected_values
                )

    def test_default_equivalent_metadata_is_accepted(self):
        source = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        option_cases = (
            {},
            {"dtype": None},
            {"dtype": torch.float32},
            {"dtype": torch.float},
            {"layout": None},
            {"layout": torch.strided},
            {"device": None},
            {"device": "cpu"},
            {"device": "cpu:0"},
            {"device": torch.device("cpu")},
            {"requires_grad": None},
            {"requires_grad": False},
            {"memory_format": torch.preserve_format},
            {
                "dtype": torch.float32,
                "layout": torch.strided,
                "device": torch.device("cpu"),
                "requires_grad": False,
                "memory_format": torch.preserve_format,
            },
        )
        for options in option_cases:
            with self.subTest(options=options):
                self.assert_zero_like(
                    source,
                    torch.zeros_like(source, **options),
                    [[0.0, 0.0], [0.0, 0.0]],
                )
        self.assert_zero_like(
            source,
            torch.zeros_like(input=source),
            [[0.0, 0.0], [0.0, 0.0]],
        )

    def test_requires_grad_creates_fresh_leaves_inside_and_outside_no_grad(self):
        source = torch.ones((2, 3), requires_grad=True)
        ordinary = torch.zeros_like(source, requires_grad=True)
        with torch.no_grad():
            no_grad = torch.zeros_like(source, requires_grad=True)

        weights = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        for label, leaf in (("ordinary", ordinary), ("no_grad", no_grad)):
            with self.subTest(label=label):
                self.assert_zero_like(
                    source,
                    leaf,
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    requires_grad=True,
                )
                (leaf * weights).sum().backward()
                self.assertEqual(leaf.grad.tolist(), weights.tolist())
                self.assertIsNone(source.grad)

    def test_callable_metadata_wildcard_import_and_pickle(self):
        function = torch.zeros_like
        self.assertTrue(callable(function))
        self.assertEqual(function.__name__, "zeros_like")
        self.assertIn("zeros_like", function.__qualname__)
        self.assertEqual(function.__module__, "torch")
        self.assertIsNone(getattr(function, "__text_signature__", None))
        with self.assertRaises(ValueError):
            inspect.signature(function)

        namespace = {}
        exec("from torch_rs import *", namespace)
        self.assertIs(namespace["zeros_like"], function)
        self.assertIn("zeros_like", torch.__all__)
        self.assertIs(pickle.loads(pickle.dumps(function)), function)

    def test_override_dispatch_and_declining_handlers(self):
        marker = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        value = Override()
        result = torch.zeros_like(
            input=value,
            dtype=torch.float32,
            memory_format=torch.preserve_format,
        )
        self.assertIs(result, marker)
        self.assertEqual(
            Override.calls,
            [
                (
                    torch.zeros_like,
                    (Override,),
                    (),
                    {
                        "input": value,
                        "dtype": torch.float32,
                        "memory_format": torch.preserve_format,
                    },
                )
            ],
        )
        for device in ("cuda", "definitely invalid device"):
            with self.subTest(kind="override-device", device=device):
                Override.calls.clear()
                result = torch.zeros_like(input=value, device=device)
                self.assertIs(result, marker)
                self.assertEqual(
                    Override.calls,
                    [
                        (
                            torch.zeros_like,
                            (Override,),
                            (),
                            {"input": value, "device": device},
                        )
                    ],
                )

        tensor = torch.tensor([1.0])
        mode_calls = []

        class HandlingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        for device in ("cuda", "definitely invalid device"):
            with self.subTest(kind="mode-device", device=device):
                mode = HandlingMode()
                with mode:
                    result = torch.zeros_like(tensor, device=device)
                self.assertIs(result, marker)
                self.assertEqual(
                    mode.calls,
                    [(torch.zeros_like, (), (tensor,), {"device": device})],
                )

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                mode_calls.append((self.label, func, types, args, kwargs))
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = torch.zeros_like(tensor)

        self.assertEqual([call[0] for call in mode_calls], ["upper", "lower"])
        self.assertTrue(all(call[1] is torch.zeros_like for call in mode_calls))
        self.assertTrue(all(call[2] == () for call in mode_calls))
        self.assertTrue(all(call[3] == (tensor,) for call in mode_calls))
        self.assertIsNone(mode_calls[0][4])
        self.assertEqual(mode_calls[1][4], {})
        self.assert_zero_like(tensor, forwarded, [0.0])

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaisesRegex(
            TypeError,
            "__torch_function__ handlers returned NotImplemented",
        ):
            torch.zeros_like(DecliningOverride())

        class DecliningMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaisesRegex(
            TypeError,
            "__torch_function__ handlers returned NotImplemented",
        ):
            with DecliningMode():
                torch.zeros_like(tensor)

    def test_unsupported_forms_are_rejected_without_source_mutation(self):
        source = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        noncontiguous = source.transpose(0, 1)
        source_before = (
            source.tolist(),
            source.stride(),
            source.storage_offset(),
            source.grad,
        )

        invalid_calls = (
            (lambda: torch.zeros_like(), TypeError, "missing"),
            (lambda: torch.zeros_like(source, source), TypeError, "positional"),
            (lambda: torch.zeros_like(source, out=source), TypeError, "out"),
            (lambda: torch.zeros_like(source, dtype=object()), TypeError, "dtype"),
            (lambda: torch.zeros_like(source, layout=object()), TypeError, "layout"),
            (lambda: torch.zeros_like(source, device=object()), TypeError, "device"),
            (lambda: torch.zeros_like(source, device="cuda"), RuntimeError, "cuda"),
            (
                lambda: torch.zeros_like(source, requires_grad=1),
                TypeError,
                "requires_grad",
            ),
            (
                lambda: torch.zeros_like(
                    source, memory_format=torch.contiguous_format
                ),
                RuntimeError,
                "memory_format=torch.preserve_format",
            ),
            (
                lambda: torch.zeros_like(source, memory_format=object()),
                TypeError,
                "memory_format",
            ),
            (
                lambda: torch.zeros_like(noncontiguous),
                NotImplementedError,
                "contiguous",
            ),
        )
        for call, error_type, message in invalid_calls:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error_type, re.escape(message)):
                    call()

        self.assertEqual(
            (source.tolist(), source.stride(), source.storage_offset(), source.grad),
            source_before,
        )

    def test_other_like_factories_remain_unsupported(self):
        namespace = {}
        exec("from torch_rs import *", namespace)
        for name in ("ones_like", "empty_like", "full_like"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch, name))
                self.assertNotIn(name, torch.__all__)
                self.assertNotIn(name, namespace)


if __name__ == "__main__":
    unittest.main()
