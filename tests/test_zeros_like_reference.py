import copy
import importlib
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
class ZerosLikeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "zeros_like differentials require pinned PyTorch 2.13.0"
            )

    def tensor_cases(self, module):
        return (
            ("scalar", module.tensor(-3.5, dtype=module.float32, requires_grad=True)),
            ("empty vector", module.zeros((0,), dtype=module.float32)),
            ("empty matrix", module.zeros((0, 3), dtype=module.float32)),
            (
                "multidimensional",
                module.tensor(
                    [[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]],
                    dtype=module.float32,
                    requires_grad=True,
                ),
            ),
            ("offset contiguous view", module.ones((3, 2), dtype=module.float32)[1]),
            (
                "singleton transpose",
                module.ones((2, 1), dtype=module.float32).transpose(0, 1),
            ),
            (
                "empty transpose",
                module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2),
            ),
        )

    def option_cases(self, module):
        return (
            {},
            {"dtype": None},
            {"dtype": module.float32},
            {"dtype": module.float},
            {"layout": None},
            {"layout": module.strided},
            {"device": None},
            {"device": "cpu"},
            {"device": "cpu:0"},
            {"device": module.device("cpu")},
            {"device": module.device("cpu", 0)},
            {"requires_grad": None},
            {"requires_grad": False},
            {"requires_grad": True},
            {"memory_format": None},
            {"memory_format": module.preserve_format},
            {"memory_format": module.contiguous_format},
            {
                "dtype": module.float32,
                "layout": module.strided,
                "device": module.device("cpu"),
                "requires_grad": True,
                "memory_format": module.preserve_format,
            },
        )

    def flat_value_bits(self, module, tensor):
        if module is reference_torch:
            array = tensor.detach().cpu().numpy()
        else:
            array = np.asarray(tensor)
        return array.reshape(-1).view(np.uint32).tolist()

    def tensor_state(self, module, tensor):
        return {
            "shape": tuple(tensor.shape),
            "stride": tensor.stride(),
            "storage_offset": tensor.storage_offset(),
            "dtype": str(tensor.dtype),
            "dtype_identity": tensor.dtype is module.float32,
            "device": str(tensor.device),
            "layout": str(tensor.layout),
            "layout_identity": tensor.layout is module.strided,
            "requires_grad": tensor.requires_grad,
            "is_leaf": tensor.is_leaf,
            "grad_is_none": tensor.grad is None,
            "values": self.flat_value_bits(module, tensor),
        }

    def zeros_like_contract(self, module, source, kwargs):
        before = self.tensor_state(module, source)
        result = module.zeros_like(source, **kwargs)
        peer = module.zeros_like(source, **kwargs)
        after = self.tensor_state(module, source)
        return {
            "result": self.tensor_state(module, result),
            "source_unchanged": before == after,
            "aliases_source": result.is_set_to(source),
            "aliases_peer": result.is_set_to(peer),
            "shares_source_data_ptr": (
                result.data_ptr() == source.data_ptr() if source.numel() else None
            ),
            "shares_peer_data_ptr": (
                result.data_ptr() == peer.data_ptr() if result.numel() else None
            ),
        }

    def test_supported_results_and_storage_match_pytorch_2_13(self):
        actual_cases = self.tensor_cases(torch)
        expected_cases = self.tensor_cases(reference_torch)
        actual_options = self.option_cases(torch)
        expected_options = self.option_cases(reference_torch)

        for (case, actual), (_, expected) in zip(
            actual_cases, expected_cases, strict=True
        ):
            for actual_kwargs, expected_kwargs in zip(
                actual_options, expected_options, strict=True
            ):
                with self.subTest(case=case, kwargs=actual_kwargs):
                    self.assertEqual(
                        self.zeros_like_contract(torch, actual, actual_kwargs),
                        self.zeros_like_contract(
                            reference_torch, expected, expected_kwargs
                        ),
                    )

    def test_no_grad_and_explicit_requires_grad_match_pytorch_2_13(self):
        def contract(module):
            source = module.ones((2, 3), dtype=module.float32, requires_grad=True)
            with module.no_grad():
                default = module.zeros_like(source)
                tracked = module.zeros_like(source, requires_grad=True)
            return (
                self.tensor_state(module, default),
                self.tensor_state(module, tracked),
            )

        self.assertEqual(contract(torch), contract(reference_torch))

    def test_requires_grad_leaf_is_independent_match_pytorch_2_13(self):
        def contract(module):
            source = module.tensor(
                [1.0, 2.0, 3.0],
                dtype=module.float32,
                requires_grad=True,
            )
            result = module.zeros_like(source, requires_grad=True)
            result.sum().backward()
            source_grad = None
            if source.grad is not None:
                source_grad = self.flat_value_bits(module, source.grad)
            return (
                self.tensor_state(module, result),
                source_grad,
                self.flat_value_bits(module, result.grad),
            )

        self.assertEqual(contract(torch), contract(reference_torch))

    def capture_error(self, call):
        with self.assertRaises(Exception) as raised:
            call()
        return type(raised.exception), str(raised.exception)

    def test_supported_entry_error_messages_match_pytorch_2_13(self):
        actual = torch.ones((2, 3), dtype=torch.float32)
        expected = reference_torch.ones((2, 3), dtype=reference_torch.float32)
        cases = (
            (
                "missing input",
                lambda module, tensor: module.zeros_like(),
                lambda module, tensor: module.zeros_like(),
            ),
            (
                "too many positional",
                lambda module, tensor: module.zeros_like(tensor, tensor),
                lambda module, tensor: module.zeros_like(tensor, tensor),
            ),
            (
                "duplicate input",
                lambda module, tensor: module.zeros_like(tensor, input=tensor),
                lambda module, tensor: module.zeros_like(tensor, input=tensor),
            ),
            (
                "out none",
                lambda module, tensor: module.zeros_like(tensor, out=None),
                lambda module, tensor: module.zeros_like(tensor, out=None),
            ),
            (
                "unexpected keyword",
                lambda module, tensor: module.zeros_like(tensor, unexpected=True),
                lambda module, tensor: module.zeros_like(tensor, unexpected=True),
            ),
            (
                "bad positional input",
                lambda module, tensor: module.zeros_like(1),
                lambda module, tensor: module.zeros_like(1),
            ),
            (
                "bad keyword input",
                lambda module, tensor: module.zeros_like(input=1),
                lambda module, tensor: module.zeros_like(input=1),
            ),
            (
                "bad dtype",
                lambda module, tensor: module.zeros_like(tensor, dtype=object()),
                lambda module, tensor: module.zeros_like(tensor, dtype=object()),
            ),
            (
                "bad layout",
                lambda module, tensor: module.zeros_like(tensor, layout=object()),
                lambda module, tensor: module.zeros_like(tensor, layout=object()),
            ),
            (
                "bad device",
                lambda module, tensor: module.zeros_like(tensor, device=object()),
                lambda module, tensor: module.zeros_like(tensor, device=object()),
            ),
            (
                "bad requires_grad",
                lambda module, tensor: module.zeros_like(tensor, requires_grad=1),
                lambda module, tensor: module.zeros_like(tensor, requires_grad=1),
            ),
            (
                "bad memory_format",
                lambda module, tensor: module.zeros_like(
                    tensor,
                    memory_format=object(),
                ),
                lambda module, tensor: module.zeros_like(
                    tensor,
                    memory_format=object(),
                ),
            ),
            (
                "out after bad dtype",
                lambda module, tensor: module.zeros_like(
                    tensor,
                    dtype=object(),
                    out=None,
                ),
                lambda module, tensor: module.zeros_like(
                    tensor,
                    dtype=object(),
                    out=None,
                ),
            ),
        )

        for case, actual_call, expected_call in cases:
            with self.subTest(case=case):
                self.assertEqual(
                    self.capture_error(lambda: actual_call(torch, actual)),
                    self.capture_error(lambda: expected_call(reference_torch, expected)),
                )

    def callable_contract(self, module):
        function = module.zeros_like
        owner = function.__reduce__()[1][0]
        wildcard_namespace = {}
        exec(f"from {module.__name__} import *", wildcard_namespace)
        try:
            inspect.signature(function)
        except Exception as error:
            signature_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
            )
        else:
            signature_error = None
        return {
            "type": type(function).__name__,
            "is_builtin": type(function) is types.BuiltinFunctionType,
            "name": function.__name__,
            "qualname": function.__qualname__,
            "module": function.__module__,
            "owner_name": owner.__name__,
            "owner_qualname": owner.__qualname__,
            "owner_module": owner.__module__.replace("torch_rs._C", "torch._C"),
            "owner_path_identity": owner is module._C._VariableFunctionsClass,
            "owner_callable_identity": owner.zeros_like is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("zeros_like"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["zeros_like"] is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_exports_copy_and_pickle_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )

        old = torch.zeros_like
        native = importlib.import_module("torch_rs._C")
        self.assertIs(importlib.reload(native), native)
        self.assertIs(native.zeros_like, old)
        self.assertIs(importlib.reload(torch), torch)
        self.assertIs(torch.zeros_like, old)


if __name__ == "__main__":
    unittest.main()
