import importlib
import inspect
import re
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class ZerosLikeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "zeros_like differentials require pinned PyTorch 2.13.0"
            )

    def tensor_contract(self, module, tensor, source):
        return {
            "shape": tuple(tensor.shape),
            "stride": tensor.stride(),
            "storage_offset": tensor.storage_offset(),
            "numel": tensor.numel(),
            "values": tensor.tolist(),
            "dtype": str(tensor.dtype),
            "dtype_identity": tensor.dtype is module.float32,
            "device": str(tensor.device),
            "layout": str(tensor.layout),
            "requires_grad": tensor.requires_grad,
            "is_leaf": tensor.is_leaf,
            "grad_is_none": tensor.grad is None,
            "output_nr": tensor.output_nr,
            "same_storage": tensor.is_set_to(source),
            "data_ptr_differs": tensor.numel() == 0
            or tensor.data_ptr() != source.data_ptr(),
        }

    def source_cases(self, module):
        return (
            module.tensor(-3.5, dtype=module.float32),
            module.zeros((2, 0, 3), dtype=module.float32),
            module.ones((1, 3), dtype=module.float32).transpose(0, 1),
            module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2),
            module.tensor(
                [[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]],
                dtype=module.float32,
            ),
        )

    def test_metadata_and_fresh_storage_match_pytorch_2_13(self):
        option_factories = (
            lambda module: {},
            lambda module: {"dtype": None},
            lambda module: {"dtype": module.float32},
            lambda module: {"layout": None},
            lambda module: {"layout": module.strided},
            lambda module: {"device": None},
            lambda module: {"device": "cpu"},
            lambda module: {"device": module.device("cpu")},
            lambda module: {"requires_grad": False},
            lambda module: {"memory_format": module.preserve_format},
            lambda module: {
                "dtype": module.float32,
                "layout": module.strided,
                "device": module.device("cpu"),
                "requires_grad": False,
                "memory_format": module.preserve_format,
            },
        )
        call_factories = (
            (
                "positional",
                lambda module, source, options: module.zeros_like(source, **options),
            ),
            (
                "keyword",
                lambda module, source, options: module.zeros_like(input=source, **options),
            ),
        )
        for actual_source, expected_source in zip(
            self.source_cases(torch),
            self.source_cases(reference_torch),
            strict=True,
        ):
            for option_factory in option_factories:
                actual_options = option_factory(torch)
                expected_options = option_factory(reference_torch)
                for call_form, call_factory in call_factories:
                    with self.subTest(
                        shape=tuple(actual_source.shape),
                        options=actual_options,
                        call_form=call_form,
                    ):
                        actual = call_factory(torch, actual_source, actual_options)
                        expected = call_factory(
                            reference_torch, expected_source, expected_options
                        )
                        self.assertEqual(
                            self.tensor_contract(torch, actual, actual_source),
                            self.tensor_contract(
                                reference_torch, expected, expected_source
                            ),
                        )

    def requires_grad_outcome(self, module):
        source = module.ones((2, 3), dtype=module.float32, requires_grad=True)
        ordinary = module.zeros_like(source, requires_grad=True)
        with module.no_grad():
            no_grad = module.zeros_like(source, requires_grad=True)
        weights = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=module.float32
        )

        outputs = []
        for leaf in (ordinary, no_grad):
            (leaf * weights).sum().backward()
            outputs.append(
                (
                    self.tensor_contract(module, leaf, source),
                    leaf.grad.tolist(),
                )
            )
        outputs.append(source.grad is None)
        return outputs

    def test_requires_grad_and_no_grad_match_pytorch_2_13(self):
        self.assertEqual(
            self.requires_grad_outcome(torch),
            self.requires_grad_outcome(reference_torch),
        )

    def callable_contract(self, module):
        function = module.zeros_like
        namespace = {}
        exec(f"from {module.__name__} import *", namespace)
        try:
            signature = ("ok", str(inspect.signature(function)))
        except Exception as error:
            signature = (
                type(error).__name__,
                re.sub(r"0x[0-9a-fA-F]+", "0x...", str(error)),
            )
        return (
            callable(function),
            function.__name__,
            function.__qualname__,
            function.__module__.replace("torch_rs", "torch"),
            getattr(function, "__text_signature__", None),
            signature,
            module.__all__.count("zeros_like"),
            namespace["zeros_like"] is function,
        )

    def test_callable_metadata_and_wildcard_import_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )

    def dispatch_outcome(self, module):
        marker = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        value = Override()
        override_result = module.zeros_like(
            input=value,
            dtype=module.float32,
            memory_format=module.preserve_format,
        )
        device_override_results = []
        for device in ("cuda", "definitely invalid device"):
            device_override_results.append(
                module.zeros_like(input=value, device=device) is marker
            )

        tensor = module.tensor([1.0], dtype=module.float32)
        mode_calls = []

        device_mode_calls = []

        class HandlingMode(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                device_mode_calls.append((func, types, args, kwargs))
                return marker

        device_mode_results = []
        for device in ("cuda", "definitely invalid device"):
            with HandlingMode():
                device_mode_results.append(
                    module.zeros_like(tensor, device=device) is marker
                )

        class Mode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                mode_calls.append((self.label, func, types, args, kwargs))
                return func(*args, **(kwargs or {}))

        with Mode("lower"):
            with Mode("upper"):
                forwarded = module.zeros_like(tensor)

        def normalize(call):
            func, types, args, kwargs = call
            return (
                func is module.zeros_like,
                tuple(dispatch_type.__name__ for dispatch_type in types),
                len(args),
                None if kwargs is None else tuple(kwargs.keys()),
            )

        return (
            override_result is marker,
            tuple(device_override_results),
            tuple(normalize(call) for call in Override.calls),
            tuple(device_mode_results),
            tuple(normalize(call) for call in device_mode_calls),
            tuple(
                (label, *normalize((func, types, args, kwargs)))
                for label, func, types, args, kwargs in mode_calls
            ),
            self.tensor_contract(module, forwarded, tensor),
        )

    def test_override_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.dispatch_outcome(torch),
            self.dispatch_outcome(reference_torch),
        )

    def test_top_level_function_import_identity_matches_package(self):
        actual = importlib.import_module("torch_rs").zeros_like
        expected = importlib.import_module("torch").zeros_like
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )


if __name__ == "__main__":
    unittest.main()
