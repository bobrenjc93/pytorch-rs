import copy
import importlib
import inspect
import pickle
import re
import types
import unittest
import warnings

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class AsArrayReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("asarray differentials require pinned PyTorch 2.13.0")

    def tensor_cases(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        produced = leaf * 2.0
        tracked = produced.transpose(0, 1)
        source = module.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ],
            dtype=module.float32,
        )
        strided = source.transpose(0, 1)
        special_bits = np.asarray(
            (0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC12345),
            dtype=np.uint32,
        )
        return (
            ("scalar", module.tensor(-0.0, dtype=module.float32)),
            ("empty view", module.zeros((2, 0, 3), dtype=module.float32)[1]),
            ("strided view", strided[1]),
            ("leaf", leaf),
            ("tracked view", tracked),
            ("special bits", module.tensor(memoryview(special_bits.view(np.float32)))),
        )

    def option_cases(self, module):
        return (
            {},
            {"dtype": None},
            {"dtype": module.float32},
            {"dtype": module.float},
            {"device": None},
            {"device": "cpu"},
            {"device": module.device("cpu")},
            {"copy": None},
            {"copy": False},
            {"requires_grad": None},
            {
                "dtype": module.float32,
                "device": module.device("cpu"),
                "copy": False,
                "requires_grad": None,
            },
        )

    def scalar_option_cases(self, module):
        return (
            {},
            {"dtype": None},
            {"dtype": module.float32},
            {"dtype": module.float},
            {"device": None},
            {"device": "cpu"},
            {"device": module.device("cpu")},
            {"copy": None},
            {"requires_grad": None},
            {
                "dtype": module.float32,
                "device": module.device("cpu"),
                "copy": None,
                "requires_grad": None,
            },
        )

    def sequence_option_cases(self, module):
        return (
            {},
            {"dtype": None},
            {"dtype": module.float32},
            {"dtype": module.float},
            {"device": None},
            {"device": "cpu"},
            {"device": module.device("cpu")},
            {"copy": None},
            {"requires_grad": None},
            {
                "dtype": module.float32,
                "device": module.device("cpu"),
                "copy": None,
                "requires_grad": None,
            },
        )

    def nested_singleton(self, value, depth, container):
        for _ in range(depth):
            value = [value] if container is list else (value,)
        return value

    def tensor_state(self, module, tensor):
        if module is reference_torch:
            values = tensor.detach().cpu().numpy().reshape(-1).view(np.uint32).tolist()
        else:
            values = np.asarray(tensor).reshape(-1).view(np.uint32).tolist()
        return {
            "shape": tuple(tensor.shape),
            "stride": tensor.stride(),
            "storage_offset": tensor.storage_offset(),
            "data_ptr": tensor.data_ptr(),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "layout": str(tensor.layout),
            "requires_grad": tensor.requires_grad,
            "is_leaf": tensor.is_leaf,
            "output_nr": tensor.output_nr,
            "values": values,
        }

    def comparable_tensor_state(self, module, tensor):
        state = self.tensor_state(module, tensor)
        state.pop("data_ptr")
        return state

    def asarray_identity_contract(self, module, tensor, options):
        before = self.tensor_state(module, tensor)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = module.asarray(tensor, **options)
        after = self.tensor_state(module, tensor)
        return {
            "same_object": result is tensor,
            "same_pointer": result.data_ptr() == before["data_ptr"],
            "same_logical_storage": result.is_set_to(tensor),
            "result_state": self.comparable_tensor_state(module, result),
            "source_unchanged": before == after,
        }

    def asarray_float_scalar_contract(self, module, value, options):
        first = module.asarray(value, **options)
        second = module.asarray(value, **options)
        return {
            "fresh_object": first is not second,
            "fresh_storage": first.data_ptr() != second.data_ptr(),
            "not_set_to_duplicate": not first.is_set_to(second),
            "state": self.comparable_tensor_state(module, first),
            "numel": first.numel(),
            "grad_is_none": first.grad is None,
        }

    def asarray_rank_contract(self, module, data):
        result = module.asarray(data)
        return {
            "shape": tuple(result.shape),
            "stride": result.stride(),
            "numel": result.numel(),
            "dtype": str(result.dtype),
            "device": str(result.device),
            "layout": str(result.layout),
            "requires_grad": result.requires_grad,
            "is_leaf": result.is_leaf,
            "output_nr": result.output_nr,
        }

    def error_observation(self, call):
        try:
            call()
        except Exception as error:
            return (type(error).__name__, str(error))
        return None

    def error_type_observation(self, call):
        try:
            call()
        except Exception as error:
            return type(error).__name__
        return None

    def asarray_float_sequence_contract(self, module, data, options, no_grad=False):
        if no_grad:
            with module.no_grad():
                first = module.asarray(data, **options)
        else:
            first = module.asarray(data, **options)
        second = module.asarray(data, **options)
        return {
            "fresh_object": first is not second,
            "fresh_storage": not first.is_set_to(second),
            "state": self.comparable_tensor_state(module, first),
            "numel": first.numel(),
            "grad_is_none": first.grad is None,
        }

    def test_identity_aliasing_and_metadata_match_pytorch_2_13(self):
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
                with self.subTest(case=case, options=actual_kwargs):
                    actual_contract = self.asarray_identity_contract(
                        torch, actual, actual_kwargs
                    )
                    expected_contract = self.asarray_identity_contract(
                        reference_torch, expected, expected_kwargs
                    )
                    self.assertEqual(actual_contract, expected_contract)

    def test_python_float_scalar_creation_matches_pytorch_2_13(self):
        values = (
            0.0,
            -0.0,
            1.25,
            -3.5,
            1e39,
            -1e39,
            float("inf"),
            float("-inf"),
            float("nan"),
        )
        actual_options = self.scalar_option_cases(torch)
        expected_options = self.scalar_option_cases(reference_torch)
        for value in values:
            for actual_kwargs, expected_kwargs in zip(
                actual_options, expected_options, strict=True
            ):
                with self.subTest(value=repr(value), options=actual_kwargs):
                    actual_contract = self.asarray_float_scalar_contract(
                        torch, value, actual_kwargs
                    )
                    expected_contract = self.asarray_float_scalar_contract(
                        reference_torch, value, expected_kwargs
                    )
                    self.assertEqual(actual_contract, expected_contract)

    def test_python_float_sequence_creation_matches_pytorch_2_13(self):
        sequence_cases = (
            ("empty list", []),
            ("empty tuple", ()),
            ("flat list", [1.25, -3.5]),
            ("flat tuple", (1.25, -3.5)),
            ("nested rectangular", [[1.0, -2.0], [3.5, 4.25]]),
            (
                "mixed tuple/list",
                ([1.0, -0.0], [float("inf"), float("-inf")]),
            ),
            (
                "special floats",
                [0.0, -0.0, float("nan"), float("inf"), float("-inf")],
            ),
        )
        actual_options = self.sequence_option_cases(torch)
        expected_options = self.sequence_option_cases(reference_torch)
        for case, data in sequence_cases:
            for actual_kwargs, expected_kwargs in zip(
                actual_options, expected_options, strict=True
            ):
                with self.subTest(case=case, options=actual_kwargs):
                    actual_contract = self.asarray_float_sequence_contract(
                        torch, data, actual_kwargs
                    )
                    expected_contract = self.asarray_float_sequence_contract(
                        reference_torch, data, expected_kwargs
                    )
                    self.assertEqual(actual_contract, expected_contract)

    def test_python_float_sequence_no_grad_matches_pytorch_2_13(self):
        data = [1.0, -0.0, float("inf")]
        self.assertEqual(
            self.asarray_float_sequence_contract(torch, data, {}, no_grad=True),
            self.asarray_float_sequence_contract(
                reference_torch, data, {}, no_grad=True
            ),
        )

    def test_recursive_and_overdeep_sequence_errors_match_pytorch_2_13(self):
        recursive_list = []
        recursive_list.append(recursive_list)
        recursive_tuple = ([],)
        recursive_tuple[0].append(recursive_tuple)
        cases = (
            ("recursive list", recursive_list),
            ("recursive tuple", recursive_tuple),
            ("overdeep list", self.nested_singleton(1.0, 129, list)),
            ("overdeep tuple", self.nested_singleton(1.0, 129, tuple)),
        )
        for case, data in cases:
            with self.subTest(case=case):
                self.assertEqual(
                    self.error_observation(lambda: torch.asarray(data)),
                    self.error_observation(lambda: reference_torch.asarray(data)),
                )

        for container in (list, tuple):
            with self.subTest(max_rank=container.__name__):
                data = self.nested_singleton(1.0, 128, container)
                self.assertEqual(
                    self.asarray_rank_contract(torch, data),
                    self.asarray_rank_contract(reference_torch, data),
                )

    def test_malformed_rectangular_sequence_error_type_matches_pytorch_2_13(self):
        data = [[1.0], [2.0, 3.0]]
        self.assertEqual(
            self.error_type_observation(lambda: torch.asarray(data)),
            self.error_type_observation(lambda: reference_torch.asarray(data)),
        )

    def test_copy_false_sequence_errors_match_pytorch_2_13(self):
        recursive_list = []
        recursive_list.append(recursive_list)
        recursive_tuple = ([],)
        recursive_tuple[0].append(recursive_tuple)
        sequence_cases = (
            ("integer list", [1]),
            ("object list", [object()]),
            ("ragged list", [[1.0], [2.0, 3.0]]),
            ("recursive list", recursive_list),
            ("overdeep list", self.nested_singleton(1.0, 129, list)),
            ("object tuple", (object(),)),
            ("ragged tuple", ((1.0,), (2.0, 3.0))),
            ("recursive tuple", recursive_tuple),
            ("overdeep tuple", self.nested_singleton(1.0, 129, tuple)),
        )
        for case, data in sequence_cases:
            with self.subTest(case=case):
                self.assertEqual(
                    self.error_observation(lambda: torch.asarray(data, copy=False)),
                    self.error_observation(
                        lambda: reference_torch.asarray(data, copy=False)
                    ),
                )

        actual_device_cases = (
            ("empty device string", {"device": ""}),
            ("unknown device string", {"device": "banana"}),
            ("unsupported cuda string", {"device": "cuda"}),
            ("indexed cpu string", {"device": "cpu:0"}),
            ("indexed cpu device", {"device": torch.device("cpu", 1)}),
        )
        expected_device_cases = (
            ("empty device string", {"device": ""}),
            ("unknown device string", {"device": "banana"}),
            ("unsupported cuda string", {"device": "cuda"}),
            ("indexed cpu string", {"device": "cpu:0"}),
            (
                "indexed cpu device",
                {"device": reference_torch.device("cpu", 1)},
            ),
        )
        for (case, actual_kwargs), (_, expected_kwargs) in zip(
            actual_device_cases, expected_device_cases, strict=True
        ):
            with self.subTest(case=case):
                self.assertEqual(
                    self.error_observation(
                        lambda actual_kwargs=actual_kwargs: torch.asarray(
                            [1.0], copy=False, **actual_kwargs
                        )
                    ),
                    self.error_observation(
                        lambda expected_kwargs=expected_kwargs: reference_torch.asarray(
                            [1.0], copy=False, **expected_kwargs
                        )
                    ),
                )

    def test_autograd_identity_aliasing_matches_pytorch_2_13(self):
        outcomes = []
        for module in (torch, reference_torch):
            leaf = module.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                dtype=module.float32,
                requires_grad=True,
            )
            source = (leaf * 3.0).transpose(0, 1)[1]
            before = self.tensor_state(module, source)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = module.asarray(source, dtype=module.float32, device="cpu")
            after = self.tensor_state(module, result)
            result.sum().backward()
            outcomes.append(
                (
                    result is source,
                    before["data_ptr"] == after["data_ptr"],
                    {key: value for key, value in before.items() if key != "data_ptr"},
                    {key: value for key, value in after.items() if key != "data_ptr"},
                    np.asarray(leaf.grad).reshape(-1).tolist(),
                    module.asarray(leaf.grad) is leaf.grad,
                )
            )

        self.assertEqual(outcomes[0], outcomes[1])

    def callable_contract(self, module):
        function = module.asarray
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
            "owner_callable_identity": owner.asarray is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("asarray"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["asarray"] is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_imports_copy_pickle_and_reload_match_pytorch_2_13(self):
        self.assertEqual(self.callable_contract(torch), self.callable_contract(reference_torch))

        old = torch.asarray
        native = torch._C
        self.assertIs(importlib.reload(native), native)
        self.assertIs(native.asarray, old)
        self.assertIs(importlib.reload(torch), torch)
        self.assertIs(torch.asarray, old)

    def mode_dispatch_observation(self, module, case):
        tensor = module.tensor([1.0], dtype=module.float32)
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        if case == "positional":
            call = lambda: module.asarray(tensor)
        elif case == "keyword":
            call = lambda: module.asarray(obj=tensor, dtype=module.float32)
        elif case == "scalar":
            data = 1.25
            call = lambda: module.asarray(data)
        elif case == "sequence":
            data = [1.0, 2.0]
            call = lambda: module.asarray(data)
        elif case == "device":
            data = [1.0]
            call = lambda: module.asarray(data, device="cuda")
        elif case == "copy":
            call = lambda: module.asarray(tensor, copy=True)
        else:
            call = lambda: module.asarray(tensor, requires_grad=True)

        mode = RecordingMode()
        with mode:
            result = call()
        function, dispatch_types, args, kwargs = mode.calls[0]

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    forwarded = module.asarray(obj=tensor, dtype=module.float32)

        class DecliningMode(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return NotImplemented

        try:
            with DecliningMode():
                module.asarray(tensor)
        except Exception as error:
            declining_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
            )
        else:
            declining_error = None

        return {
            "intercepted": result is marker,
            "call_count": len(mode.calls),
            "function_name": function.__name__,
            "function_qualname": function.__qualname__,
            "function_identity": function is module.asarray,
            "types_empty": dispatch_types == (),
            "args_len": len(args),
            "kwargs_keys": None if kwargs is None else tuple(kwargs.keys()),
            "forwarding_order": order,
            "forwarded_is_tensor": forwarded is tensor,
            "declining_error": declining_error,
            "stack_depth": len(module.overrides._get_current_function_mode_stack()),
        }

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        for case in (
            "positional",
            "keyword",
            "scalar",
            "sequence",
            "device",
            "copy",
            "requires_grad",
        ):
            with self.subTest(case=case):
                self.assertEqual(
                    self.mode_dispatch_observation(torch, case),
                    self.mode_dispatch_observation(reference_torch, case),
                )

    def test_binding_errors_match_pytorch_2_13_for_exposed_schema(self):
        actual = torch.tensor([1.0], dtype=torch.float32)
        expected = reference_torch.tensor([1.0], dtype=reference_torch.float32)
        cases = (
            (lambda: torch.asarray(), lambda: reference_torch.asarray()),
            (
                lambda: torch.asarray(actual, actual),
                lambda: reference_torch.asarray(expected, expected),
            ),
            (
                lambda: torch.asarray(actual, obj=actual),
                lambda: reference_torch.asarray(expected, obj=expected),
            ),
            (
                lambda: torch.asarray(data=actual),
                lambda: reference_torch.asarray(data=expected),
            ),
            (
                lambda: torch.asarray(actual, data=actual),
                lambda: reference_torch.asarray(expected, data=expected),
            ),
            (
                lambda: torch.asarray(actual, out=None),
                lambda: reference_torch.asarray(expected, out=None),
            ),
            (
                lambda: torch.asarray(actual, pin_memory=False),
                lambda: reference_torch.asarray(expected, pin_memory=False),
            ),
            (
                lambda: torch.asarray(actual, copy=0),
                lambda: reference_torch.asarray(expected, copy=0),
            ),
            (
                lambda: torch.asarray(actual, requires_grad=0),
                lambda: reference_torch.asarray(expected, requires_grad=0),
            ),
            (
                lambda: torch.asarray(actual, dtype=1),
                lambda: reference_torch.asarray(expected, dtype=1),
            ),
            (
                lambda: torch.asarray(actual, device=1.5),
                lambda: reference_torch.asarray(expected, device=1.5),
            ),
            (
                lambda: torch.asarray(actual, device=""),
                lambda: reference_torch.asarray(expected, device=""),
            ),
            (
                lambda: torch.asarray(actual, device="banana"),
                lambda: reference_torch.asarray(expected, device="banana"),
            ),
        )
        for actual_call, expected_call in cases:
            with self.subTest(case=actual_call):
                with self.assertRaises(Exception) as actual_raised:
                    actual_call()
                with self.assertRaises(Exception) as expected_raised:
                    expected_call()
                self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
                self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))


if __name__ == "__main__":
    unittest.main()
