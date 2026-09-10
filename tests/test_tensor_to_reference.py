import inspect
import types
import unittest

import numpy as np
import torch_rs as torch
import torch_rs.nn.functional as functional

try:
    import torch as reference_torch
    import torch.nn.functional as reference_functional
except ImportError:
    reference_torch = None
    reference_functional = None


METHOD_DOC = """
to(*args, **kwargs) -> Tensor

Converts an exact native ``float32`` Tensor to equivalent supported metadata or
storage. CPU requests that leave dtype and device unchanged return ``self``
unless ``copy=True`` or an indexed CPU device such as ``"cpu:0"`` is requested;
CPU copy requests return a fresh Tensor and record ``ToCopyBackward0`` when
autograd is active. CPU tensors without autograd support synchronized copies
to explicit CUDA devices, preserving dense strides and packing sparse views.
Native CUDA tensors support synchronized transfer to CPU. Contiguous rank-1
CUDA tensors without autograd also support same-device ``copy=True``;
ordinary same-device requests return ``self``.

Supported forms include ``to()``, ``to(torch.float32)``, ``to(torch.float)``,
``to("cpu")``, ``to(torch.device("cpu"))``, ``to(device="cpu")``,
``to("cpu", torch.float32)``, ``to(device="cpu", dtype=torch.float32)``,
``to("cuda:0")``, ``to(device=torch.device("cuda", 0))``, and
``to(other)`` when ``other`` is another exact native ``float32`` Tensor on CPU
or CUDA. ``copy`` may be ``True`` or ``False``;
``non_blocking`` must be ``False``; ``memory_format`` may be omitted, ``None``,
or ``torch.preserve_format``.

Unsupported: dtype-changing conversions such as ``torch.float64``, autograd
through CUDA transfers, cross-device CUDA copies, unindexed CUDA targets, devices
other than CPU and CUDA, ``non_blocking=True``, memory formats other than
``torch.preserve_format``, Tensor subclasses, and non-native tensors.

Example::

    >>> tensor = torch.tensor([1.0], dtype=torch.float32)
    >>> tensor.to(torch.float32) is tensor
    True
    >>> tensor.to(copy=True) is tensor
    False
"""


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorToReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("Tensor.to() differentials require pinned PyTorch 2.13.0")

    def tensor_cases(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        source = module.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ],
            dtype=module.float32,
        )
        return (
            module.tensor(-0.0, dtype=module.float32),
            module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
            source.transpose(0, 1)[1],
            leaf,
            tracked,
        )

    @staticmethod
    def value_bits(tensor):
        if tensor.numel() == 0:
            return ()
        values = np.ascontiguousarray(np.asarray(tensor.detach()))
        return tuple(values.reshape(-1).view(np.uint32).tolist())

    def tensor_contract(self, result, source):
        return {
            "result_is_source": result is source,
            "result_is_set_to_source": result.is_set_to(source),
            "pointer_matches_source": result.data_ptr() == source.data_ptr(),
            "shape": tuple(result.shape),
            "stride": tuple(result.stride()),
            "storage_offset": result.storage_offset(),
            "dtype": str(result.dtype),
            "device": str(result.device),
            "requires_grad": result.requires_grad,
            "is_leaf": result.is_leaf,
            "bits": self.value_bits(result),
        }

    def identity_calls(self, module):
        other = module.tensor([7.0], dtype=module.float32)
        return (
            ("no args", lambda tensor: tensor.to()),
            ("dtype positional", lambda tensor: tensor.to(module.float32)),
            ("float alias positional", lambda tensor: tensor.to(module.float)),
            ("dtype keyword", lambda tensor: tensor.to(dtype=module.float32)),
            ("device string", lambda tensor: tensor.to("cpu")),
            ("device object", lambda tensor: tensor.to(module.device("cpu"))),
            ("device keyword", lambda tensor: tensor.to(device="cpu")),
            ("device dtype positional", lambda tensor: tensor.to("cpu", module.float32)),
            (
                "device dtype keyword",
                lambda tensor: tensor.to(device="cpu", dtype=module.float),
            ),
            ("other tensor", lambda tensor: tensor.to(other)),
            ("nonblocking false", lambda tensor: tensor.to(non_blocking=False)),
            ("copy false", lambda tensor: tensor.to(copy=False)),
            ("memory format none", lambda tensor: tensor.to(memory_format=None)),
            (
                "memory format preserve",
                lambda tensor: tensor.to(memory_format=module.preserve_format),
            ),
        )

    def clone_calls(self, module):
        other = module.tensor([7.0], dtype=module.float32)
        return (
            ("copy keyword", lambda tensor: tensor.to(copy=True)),
            ("dtype copy keyword", lambda tensor: tensor.to(module.float32, copy=True)),
            ("device copy keyword", lambda tensor: tensor.to("cpu", copy=True)),
            (
                "device dtype positional copy",
                lambda tensor: tensor.to("cpu", module.float32, False, True),
            ),
            ("other copy keyword", lambda tensor: tensor.to(other, copy=True)),
            ("indexed cpu string", lambda tensor: tensor.to("cpu:0")),
            ("indexed cpu device", lambda tensor: tensor.to(module.device("cpu:0"))),
            ("indexed cpu keyword", lambda tensor: tensor.to(device="cpu:1")),
        )

    def test_identity_forms_match_pytorch_2_13(self):
        actual_cases = self.tensor_cases(torch)
        expected_cases = self.tensor_cases(reference_torch)
        actual_calls = self.identity_calls(torch)
        expected_calls = self.identity_calls(reference_torch)
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            for (actual_name, actual_call), (expected_name, expected_call) in zip(
                actual_calls, expected_calls, strict=True
            ):
                self.assertEqual(actual_name, expected_name)
                with self.subTest(case=case, call=actual_name):
                    self.assertEqual(
                        self.tensor_contract(actual_call(actual), actual),
                        self.tensor_contract(expected_call(expected), expected),
                    )

    def test_copy_and_indexed_cpu_forms_match_pytorch_2_13(self):
        actual_cases = self.tensor_cases(torch)
        expected_cases = self.tensor_cases(reference_torch)
        actual_calls = self.clone_calls(torch)
        expected_calls = self.clone_calls(reference_torch)
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            for (actual_name, actual_call), (expected_name, expected_call) in zip(
                actual_calls, expected_calls, strict=True
            ):
                self.assertEqual(actual_name, expected_name)
                with self.subTest(case=case, call=actual_name):
                    self.assertEqual(
                        self.tensor_contract(actual_call(actual), actual),
                        self.tensor_contract(expected_call(expected), expected),
                    )

    def autograd_copy_outcome(self, module):
        leaf = module.ones((2, 3), dtype=module.float32, requires_grad=True)
        source = (leaf * 3.0).transpose(0, 1)
        copied = source.to(copy=True)
        copied.sum().backward()
        return {
            "copied": self.tensor_contract(copied, source),
            "gradient": self.value_bits(leaf.grad),
        }

    def no_grad_copy_outcome(self, module):
        leaf = module.ones((2, 3), dtype=module.float32, requires_grad=True)
        source = (leaf * 4.0).transpose(0, 1)
        with module.no_grad():
            copied = source.to(copy=True)
        return {
            "copied": self.tensor_contract(copied, source),
            "source_requires_grad": source.requires_grad,
            "grad_mode_restored": module.is_grad_enabled(),
        }

    def no_grad_identity_outcome(self, module):
        leaf = module.ones((2, 3), dtype=module.float32, requires_grad=True)
        source = (leaf * 5.0).transpose(0, 1)
        with module.no_grad():
            result = source.to()
        result.sum().backward()
        return {
            "identity": self.tensor_contract(result, source),
            "gradient": self.value_bits(leaf.grad),
            "grad_mode_restored": module.is_grad_enabled(),
        }

    def test_grad_and_no_grad_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.autograd_copy_outcome(torch),
            self.autograd_copy_outcome(reference_torch),
        )
        self.assertEqual(
            self.no_grad_copy_outcome(torch),
            self.no_grad_copy_outcome(reference_torch),
        )
        self.assertEqual(
            self.no_grad_identity_outcome(torch),
            self.no_grad_identity_outcome(reference_torch),
        )

    def copy_grad_fn_error_outcome(self, module, functional_module):
        source = module.tensor([2.0], dtype=module.float32, requires_grad=True) + 0.0
        probability = source.to(copy=True)
        try:
            functional_module.dropout(
                module.tensor([1.0], dtype=module.float32), p=probability
            )
        except Exception as error:
            return type(error).__name__, str(error)
        return "ok", None

    def test_copy_true_grad_fn_metadata_matches_pytorch_2_13(self):
        self.assertEqual(
            self.copy_grad_fn_error_outcome(torch, functional),
            self.copy_grad_fn_error_outcome(reference_torch, reference_functional),
        )

    def serialize_to_call_value(self, module, tensor, override, value):
        if value is tensor:
            return "tensor"
        if value is override:
            return "override"
        if isinstance(value, str):
            return ("str", value)
        if value is None:
            return None
        if value is module.float32:
            return "float32"
        if value is module.channels_last:
            return "channels_last"
        return repr(value)

    def serialize_to_dispatch_call(self, module, tensor, override, call):
        function, dispatch_types, args, kwargs = call
        return {
            "function_is_descriptor": function is inspect.getattr_static(module.Tensor, "to"),
            "dispatch_type_names": tuple(
                dispatch_type.__name__ for dispatch_type in dispatch_types
            ),
            "args": tuple(
                self.serialize_to_call_value(module, tensor, override, value)
                for value in args
            ),
            "kwargs": None
            if kwargs is None
            else tuple(
                (
                    key,
                    self.serialize_to_call_value(module, tensor, override, value),
                )
                for key, value in kwargs.items()
            ),
        }

    def torch_function_mode_contract(self, module):
        tensor = module.tensor([1.0], dtype=module.float32)
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        outcomes = []
        for name, call in (
            ("no args", lambda: tensor.to()),
            ("cuda", lambda: tensor.to("cuda")),
            (
                "unsupported memory format",
                lambda: tensor.to(memory_format=module.channels_last),
            ),
        ):
            mode = RecordingMode(marker)
            with mode:
                result = call()
            outcomes.append(
                (
                    name,
                    result is marker,
                    tuple(
                        self.serialize_to_dispatch_call(module, tensor, None, dispatch_call)
                        for dispatch_call in mode.calls
                    ),
                )
            )
        return tuple(outcomes)

    def torch_function_override_contract(self, module):
        tensor = module.tensor([1.0], dtype=module.float32)
        marker = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        outcomes = []
        for name, call in (
            ("first positional", lambda value: tensor.to(value)),
            ("first dtype keyword", lambda value: tensor.to(value, dtype=module.float32)),
            ("first dtype positional", lambda value: tensor.to(value, module.float32)),
            ("first none positional", lambda value: tensor.to(value, None)),
            ("cuda dtype positional", lambda value: tensor.to("cuda", value)),
            ("dtype keyword", lambda value: tensor.to(dtype=value)),
            ("copy keyword", lambda value: tensor.to(copy=value)),
            ("memory format keyword", lambda value: tensor.to(memory_format=value)),
        ):
            value = Override()
            Override.calls.clear()
            result = call(value)
            outcomes.append(
                (
                    name,
                    result is marker,
                    tuple(
                        self.serialize_to_dispatch_call(module, tensor, value, dispatch_call)
                        for dispatch_call in Override.calls
                    ),
                )
            )
        return tuple(outcomes)

    def test_torch_function_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.torch_function_mode_contract(torch),
            self.torch_function_mode_contract(reference_torch),
        )
        self.assertEqual(
            self.torch_function_override_contract(torch),
            self.torch_function_override_contract(reference_torch),
        )

    def device_string_subclass_contract(self, module):
        tensor = module.tensor([1.0], dtype=module.float32)

        class DeviceString(str):
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return "override"

        outcomes = []
        for specification in ("cpu", "cpu:0"):
            DeviceString.calls.clear()
            result = tensor.to(device=DeviceString(specification))
            outcomes.append(
                (
                    specification,
                    len(DeviceString.calls),
                    self.tensor_contract(result, tensor),
                )
            )
        return tuple(outcomes)

    def test_device_string_subclasses_parse_as_devices_without_dispatch(self):
        self.assertEqual(
            self.device_string_subclass_contract(torch),
            self.device_string_subclass_contract(reference_torch),
        )

    def device_like_override_outside_device_slot_contract(self, module):
        tensor = module.tensor([1.0], dtype=module.float32)
        marker = object()

        class DeviceString(str):
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        class DeviceOrdinal(int):
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        outcomes = []
        for name, value, call in (
            (
                "dtype positional string",
                DeviceString("dtype"),
                lambda value: tensor.to("cuda", value),
            ),
            (
                "copy keyword string",
                DeviceString("copy"),
                lambda value: tensor.to("cuda", copy=value),
            ),
            (
                "non_blocking keyword int",
                DeviceOrdinal(0),
                lambda value: tensor.to("cuda", non_blocking=value),
            ),
        ):
            type(value).calls.clear()
            result = call(value)
            outcomes.append(
                (
                    name,
                    result is marker,
                    tuple(
                        self.serialize_to_dispatch_call(module, tensor, value, dispatch_call)
                        for dispatch_call in type(value).calls
                    ),
                )
            )
        return tuple(outcomes)

    def test_device_like_overrides_outside_device_slots_match_pytorch_2_13(self):
        self.assertEqual(
            self.device_like_override_outside_device_slot_contract(torch),
            self.device_like_override_outside_device_slot_contract(reference_torch),
        )

    def descriptor_lookup_contract(self, module):
        tensor = module.tensor([1.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "to")

        def run_case(name, call, fail_after=None, attribute_error_at=None):
            marker = object()
            events = []

            class FlakyDescriptor:
                def __get__(self, obj, objtype=None):
                    events.append("get")
                    lookup_count = events.count("get")
                    if attribute_error_at is not None and lookup_count == attribute_error_at:
                        raise AttributeError("transient __torch_function__ miss")
                    if fail_after is not None and lookup_count > fail_after:
                        raise RuntimeError("late override failure")

                    def handler(func, types, args=(), kwargs=None):
                        events.append(
                            (
                                "call",
                                func is descriptor,
                                tuple(dispatch_type.__name__ for dispatch_type in types),
                                len(args),
                                None if kwargs is None else tuple(kwargs),
                            )
                        )
                        return marker

                    return handler

            class Override:
                __torch_function__ = FlakyDescriptor()

            try:
                result = call(tensor, Override())
            except Exception as error:
                outcome = ("error", type(error).__name__)
            else:
                outcome = ("ok", result is marker)
            return name, outcome, tuple(events)

        return tuple(
            run_case(name, call, **options)
            for name, call, options in (
                (
                    "first positional late failure after two lookups",
                    lambda tensor, value: tensor.to(value),
                    {"fail_after": 2},
                ),
                (
                    "unsupported device before override late failure after two lookups",
                    lambda tensor, value: tensor.to("cuda", value),
                    {"fail_after": 2},
                ),
                (
                    "copy keyword late failure after two lookups",
                    lambda tensor, value: tensor.to(copy=value),
                    {"fail_after": 2},
                ),
                (
                    "first positional transient miss",
                    lambda tensor, value: tensor.to(value),
                    {"attribute_error_at": 1},
                ),
                (
                    "dtype keyword transient miss",
                    lambda tensor, value: tensor.to(dtype=value),
                    {"attribute_error_at": 1},
                ),
                (
                    "positional dtype after device transient miss",
                    lambda tensor, value: tensor.to("cuda", value),
                    {"attribute_error_at": 1},
                ),
                (
                    "device keyword transient miss",
                    lambda tensor, value: tensor.to(device=value),
                    {"attribute_error_at": 1},
                ),
                (
                    "copy keyword transient miss",
                    lambda tensor, value: tensor.to(copy=value),
                    {"attribute_error_at": 1},
                ),
                (
                    "non_blocking keyword transient miss",
                    lambda tensor, value: tensor.to(non_blocking=value),
                    {"attribute_error_at": 1},
                ),
                (
                    "memory_format keyword transient miss",
                    lambda tensor, value: tensor.to(memory_format=value),
                    {"attribute_error_at": 1},
                ),
            )
        )

    def test_torch_function_descriptor_lookup_counts_match_pytorch_2_13(self):
        self.assertEqual(
            self.descriptor_lookup_contract(torch),
            self.descriptor_lookup_contract(reference_torch),
        )

    def callable_contract(self, module):
        tensor = module.tensor([1.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "to")
        bound = tensor.to

        def signature_outcome(callable_object):
            try:
                return "signature", str(inspect.signature(callable_object))
            except Exception as error:
                return "error", type(error).__name__

        return {
            "descriptor_type": type(descriptor).__name__,
            "bound_type": type(bound).__name__,
            "descriptor_repr": repr(descriptor),
            "descriptor_name": descriptor.__name__,
            "descriptor_qualname": descriptor.__qualname__,
            "bound_name": bound.__name__,
            "bound_qualname": bound.__qualname__,
            "doc": descriptor.__doc__,
            "bound_doc": bound.__doc__,
            "descriptor_text_signature": descriptor.__text_signature__,
            "bound_text_signature": bound.__text_signature__,
            "signatures": (
                signature_outcome(descriptor),
                signature_outcome(bound),
            ),
            "owner_name": descriptor.__objclass__.__name__,
            "owner_module": descriptor.__objclass__.__module__,
            "descriptor_has_module": hasattr(descriptor, "__module__"),
            "bound_module": bound.__module__,
            "descriptor_result_is_receiver": descriptor(tensor) is tensor,
            "bound_result_is_receiver": bound() is tensor,
            "types_match": (
                type(descriptor) is types.MethodDescriptorType,
                type(bound) is types.BuiltinMethodType,
            ),
        }

    def test_descriptor_documentation_and_binding_match_pytorch_2_13(self):
        actual = self.callable_contract(torch)
        expected = self.callable_contract(reference_torch)

        self.assertEqual(actual.pop("doc"), METHOD_DOC)
        self.assertEqual(actual.pop("bound_doc"), METHOD_DOC)
        expected.pop("doc")
        expected.pop("bound_doc")
        self.assertEqual(actual, expected)

    def test_unsupported_conversions_fail_closed(self):
        tensor = torch.tensor([1.0], dtype=torch.float32)
        unsupported = (
            lambda: tensor.to(reference_torch.float64),
            lambda: tensor.to(dtype=reference_torch.float64),
            lambda: tensor.to("cuda"),
            lambda: tensor.to("meta"),
            lambda: tensor.to(memory_format=torch.contiguous_format),
            lambda: tensor.to(memory_format=torch.channels_last),
            lambda: tensor.to(non_blocking=True),
        )
        for call in unsupported:
            with self.subTest(call=call):
                with self.assertRaises((TypeError, RuntimeError, NotImplementedError)):
                    call()


if __name__ == "__main__":
    unittest.main()
