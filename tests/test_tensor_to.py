import inspect
import re
import types
import unittest

import numpy as np
import torch_rs as torch
import torch_rs.nn.functional as functional


METHOD_DOC = """
to(*args, **kwargs) -> Tensor

Converts an exact native ``float32`` Tensor to equivalent supported metadata or
storage. CPU requests that leave dtype and device unchanged return ``self``
unless ``copy=True`` or an indexed CPU device such as ``"cpu:0"`` is requested;
CPU copy requests return a fresh Tensor and record ``ToCopyBackward0`` when
autograd is active. CUDA tensors created by the public 1-D float32
``torch.zeros`` path support synchronized transfer to CPU.

Supported forms include ``to()``, ``to(torch.float32)``, ``to(torch.float)``,
``to("cpu")``, ``to(torch.device("cpu"))``, ``to(device="cpu")``,
``to("cpu", torch.float32)``, ``to(device="cpu", dtype=torch.float32)``, and
``to(other)`` when ``other`` is another exact native ``float32`` Tensor on CPU
or the same narrow CUDA storage path. ``copy`` may be ``True`` or ``False``;
``non_blocking`` must be ``False``; ``memory_format`` may be omitted, ``None``,
or ``torch.preserve_format``.

Unsupported: dtype-changing conversions such as ``torch.float64``, CPU-to-CUDA
transfers, CUDA-to-CUDA copies, devices other than CPU and the narrow CUDA
zero-tensor storage path, ``non_blocking=True``, memory formats other than
``torch.preserve_format``, Tensor subclasses, and non-native tensors.

Example::

    >>> tensor = torch.tensor([1.0], dtype=torch.float32)
    >>> tensor.to(torch.float32) is tensor
    True
    >>> tensor.to(copy=True) is tensor
    False
"""


class TensorToTests(unittest.TestCase):
    def tensor_cases(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        source = torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ],
            dtype=torch.float32,
        )
        offset_noncontiguous = source.transpose(0, 1)[1]
        empty_view = torch.zeros((2, 0, 3), dtype=torch.float32).transpose(0, 2)[1]

        self.assertFalse(offset_noncontiguous.is_contiguous())
        self.assertGreater(offset_noncontiguous.storage_offset(), 0)
        return (
            ("scalar", torch.tensor(-0.0, dtype=torch.float32)),
            ("empty view", empty_view),
            ("offset noncontiguous view", offset_noncontiguous),
            ("autograd leaf", leaf),
            ("autograd non-leaf", tracked),
        )

    @staticmethod
    def value_bits(tensor):
        if tensor.numel() == 0:
            return ()
        return tuple(np.asarray(tensor.detach()).reshape(-1).view(np.uint32).tolist())

    @staticmethod
    def identity_calls():
        other = torch.tensor([7.0], dtype=torch.float32)
        return (
            ("no args", lambda tensor: tensor.to()),
            ("dtype positional", lambda tensor: tensor.to(torch.float32)),
            ("float alias positional", lambda tensor: tensor.to(torch.float)),
            ("dtype keyword", lambda tensor: tensor.to(dtype=torch.float32)),
            ("device string", lambda tensor: tensor.to("cpu")),
            ("device object", lambda tensor: tensor.to(torch.device("cpu"))),
            ("device keyword", lambda tensor: tensor.to(device="cpu")),
            (
                "device dtype positional",
                lambda tensor: tensor.to("cpu", torch.float32),
            ),
            (
                "device dtype keyword",
                lambda tensor: tensor.to(device="cpu", dtype=torch.float),
            ),
            ("other tensor", lambda tensor: tensor.to(other)),
            ("nonblocking false", lambda tensor: tensor.to(non_blocking=False)),
            ("copy false", lambda tensor: tensor.to(copy=False)),
            ("memory format none", lambda tensor: tensor.to(memory_format=None)),
            (
                "memory format preserve",
                lambda tensor: tensor.to(memory_format=torch.preserve_format),
            ),
        )

    @staticmethod
    def clone_calls():
        other = torch.tensor([7.0], dtype=torch.float32)
        return (
            ("copy keyword", lambda tensor: tensor.to(copy=True)),
            ("dtype copy keyword", lambda tensor: tensor.to(torch.float32, copy=True)),
            ("device copy keyword", lambda tensor: tensor.to("cpu", copy=True)),
            ("device dtype positional copy", lambda tensor: tensor.to("cpu", torch.float32, False, True)),
            ("other copy keyword", lambda tensor: tensor.to(other, copy=True)),
            ("indexed cpu string", lambda tensor: tensor.to("cpu:0")),
            ("indexed cpu device", lambda tensor: tensor.to(torch.device("cpu:0"))),
            ("indexed cpu keyword", lambda tensor: tensor.to(device="cpu:1")),
        )

    def assert_identity(self, tensor, result):
        metadata = (
            tensor.shape,
            tensor.stride(),
            tensor.storage_offset(),
            tensor.data_ptr(),
            tensor.dtype,
            tensor.device,
            tensor.requires_grad,
            tensor.is_leaf,
        )
        bits = self.value_bits(tensor)

        self.assertIs(result, tensor)
        self.assertTrue(result.is_set_to(tensor))
        self.assertEqual(
            (
                result.shape,
                result.stride(),
                result.storage_offset(),
                result.data_ptr(),
                result.dtype,
                result.device,
                result.requires_grad,
                result.is_leaf,
            ),
            metadata,
        )
        self.assertEqual(self.value_bits(result), bits)

    def assert_fresh_clone(self, tensor, result):
        expected = tensor.clone()
        self.assertIsNot(result, tensor)
        self.assertFalse(result.is_set_to(tensor))
        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result.stride(), expected.stride())
        self.assertEqual(result.storage_offset(), expected.storage_offset())
        self.assertIs(result.dtype, tensor.dtype)
        self.assertEqual(result.device, tensor.device)
        self.assertEqual(result.requires_grad, expected.requires_grad)
        self.assertEqual(result.is_leaf, expected.is_leaf)
        self.assertEqual(self.value_bits(result), self.value_bits(tensor))

    def test_default_equivalent_requests_return_exact_receiver(self):
        for case, tensor in self.tensor_cases():
            for call_name, call in self.identity_calls():
                with self.subTest(case=case, call=call_name):
                    self.assert_identity(tensor, call(tensor))

    def test_copy_and_indexed_cpu_requests_materialize_fresh_clones(self):
        for case, tensor in self.tensor_cases():
            for call_name, call in self.clone_calls():
                with self.subTest(case=case, call=call_name):
                    self.assert_fresh_clone(tensor, call(tensor))

    def test_copy_true_preserves_autograd_and_no_grad_behavior(self):
        leaf = torch.ones((2, 3), dtype=torch.float32, requires_grad=True)
        source = (leaf * 3.0).transpose(0, 1)
        copied = source.to(copy=True)
        self.assertTrue(copied.requires_grad)
        self.assertFalse(copied.is_leaf)
        self.assertFalse(copied.is_set_to(source))
        copied.sum().backward()
        np.testing.assert_array_equal(
            np.asarray(leaf.grad), np.full((2, 3), 3.0, dtype=np.float32)
        )

        leaf = torch.ones((2, 3), dtype=torch.float32, requires_grad=True)
        source = (leaf * 4.0).transpose(0, 1)
        with torch.no_grad():
            copied = source.to(copy=True)
        self.assertFalse(copied.requires_grad)
        self.assertTrue(copied.is_leaf)
        self.assertFalse(copied.is_set_to(source))
        self.assertTrue(source.requires_grad)

        leaf = torch.ones((2, 3), dtype=torch.float32, requires_grad=True)
        source = (leaf * 5.0).transpose(0, 1)
        with torch.no_grad():
            identity = source.to()
        self.assertIs(identity, source)
        self.assertTrue(identity.requires_grad)
        identity.sum().backward()
        np.testing.assert_array_equal(
            np.asarray(leaf.grad), np.full((2, 3), 5.0, dtype=np.float32)
        )

    def test_copy_true_records_to_copy_autograd_node_in_public_errors(self):
        source = torch.tensor([2.0], dtype=torch.float32, requires_grad=True) + 0.0
        probability = source.to(copy=True)

        with self.assertRaisesRegex(
            ValueError,
            r"dropout probability has to be between 0 and 1, but got "
            r"tensor\(\[2\.\], grad_fn=<ToCopyBackward0>\)",
        ):
            functional.dropout(torch.tensor([1.0], dtype=torch.float32), p=probability)

    def test_unsupported_forms_fail_closed(self):
        tensor = torch.tensor([1.0], dtype=torch.float32)
        unsupported = (
            (
                lambda: tensor.to("cuda"),
                NotImplementedError,
                "unindexed CUDA devices are not supported",
            ),
            (lambda: tensor.to("cuda:0"), RuntimeError, "only 'cpu' is implemented"),
            (lambda: tensor.to("meta"), RuntimeError, "only 'cpu' is implemented"),
            (
                lambda: tensor.to(non_blocking=True),
                NotImplementedError,
                "non_blocking=True is not supported",
            ),
            (
                lambda: tensor.to(torch.float32, True),
                NotImplementedError,
                "non_blocking=True is not supported",
            ),
            (
                lambda: tensor.to(memory_format=torch.contiguous_format),
                NotImplementedError,
                "only torch.preserve_format memory_format is supported",
            ),
            (
                lambda: tensor.to(memory_format=torch.channels_last),
                NotImplementedError,
                "only torch.preserve_format memory_format is supported",
            ),
            (lambda: tensor.to(copy=1), TypeError, "invalid combination"),
            (lambda: tensor.to(dtype=object()), TypeError, "invalid combination"),
        )
        for call, error_type, message in unsupported:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error_type, re.escape(message)):
                    call()

        with self.assertRaisesRegex(TypeError, "not an acceptable base type"):
            type("TensorSubclass", (torch.Tensor,), {})

    def test_device_subclasses_with_torch_function_parse_as_native_devices(self):
        tensor = torch.tensor([1.0], dtype=torch.float32)
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

        DeviceString.calls.clear()
        self.assert_identity(tensor, tensor.to(device=DeviceString("cpu")))
        self.assertEqual(DeviceString.calls, [])

        DeviceString.calls.clear()
        self.assert_fresh_clone(tensor, tensor.to(device=DeviceString("cpu:0")))
        self.assertEqual(DeviceString.calls, [])

        DeviceString.calls.clear()
        with self.assertRaisesRegex(
            NotImplementedError, "unindexed CUDA devices are not supported"
        ):
            tensor.to(device=DeviceString("cuda"))
        self.assertEqual(DeviceString.calls, [])

        for call in (
            lambda: tensor.to(DeviceOrdinal(0)),
            lambda: tensor.to(device=DeviceOrdinal(0)),
        ):
            DeviceOrdinal.calls.clear()
            with self.subTest(call=call):
                with self.assertRaisesRegex(
                    NotImplementedError, "CUDA device ordinals are not supported"
                ):
                    call()
                self.assertEqual(DeviceOrdinal.calls, [])

    def test_device_like_torch_function_overrides_outside_device_slots_dispatch(self):
        tensor = torch.tensor([1.0], dtype=torch.float32)
        descriptor = inspect.getattr_static(torch.Tensor, "to")
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

        def assert_dispatch(value, call, expected_tail, expected_kwargs):
            type(value).calls.clear()
            self.assertIs(call(value), marker)
            self.assertEqual(len(type(value).calls), 1)

            function, dispatch_types, args, kwargs = type(value).calls[0]
            self.assertIs(function, descriptor)
            self.assertEqual(dispatch_types, (type(value),))
            self.assertIs(args[0], tensor)
            self.assertEqual(len(args), len(expected_tail) + 1)
            for actual, expected in zip(args[1:], expected_tail, strict=True):
                if expected is value:
                    self.assertIs(actual, expected)
                else:
                    self.assertEqual(actual, expected)
            if expected_kwargs is None:
                self.assertIsNone(kwargs)
            else:
                self.assertEqual(tuple(kwargs), tuple(expected_kwargs))
                for key, expected in expected_kwargs.items():
                    if expected is value:
                        self.assertIs(kwargs[key], expected)
                    else:
                        self.assertEqual(kwargs[key], expected)

        dtype = DeviceString("dtype")
        assert_dispatch(
            dtype,
            lambda value: tensor.to("cuda", value),
            ("cuda", dtype),
            None,
        )

        copy = DeviceString("copy")
        assert_dispatch(
            copy,
            lambda value: tensor.to("cuda", copy=value),
            ("cuda",),
            {"copy": copy},
        )

        non_blocking = DeviceOrdinal(0)
        assert_dispatch(
            non_blocking,
            lambda value: tensor.to("cuda", non_blocking=value),
            ("cuda",),
            {"non_blocking": non_blocking},
        )

    def test_torch_function_override_descriptor_lookup_is_not_prescanned(self):
        tensor = torch.tensor([1.0], dtype=torch.float32)
        descriptor = inspect.getattr_static(torch.Tensor, "to")

        def assert_dispatches_with_two_lookups(call, expected_arg_count, expected_kwargs):
            marker = object()
            events = []

            class LookupLimitedDescriptor:
                def __get__(self, obj, objtype=None):
                    events.append("get")
                    if events.count("get") > 2:
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
                __torch_function__ = LookupLimitedDescriptor()

            value = Override()
            self.assertIs(call(value), marker)
            self.assertEqual(
                events,
                [
                    "get",
                    "get",
                    (
                        "call",
                        True,
                        ("Override",),
                        expected_arg_count,
                        None if expected_kwargs is None else tuple(expected_kwargs),
                    ),
                ],
            )

        assert_dispatches_with_two_lookups(
            lambda value: tensor.to(value),
            2,
            None,
        )
        assert_dispatches_with_two_lookups(
            lambda value: tensor.to("cuda", value),
            3,
            None,
        )
        assert_dispatches_with_two_lookups(
            lambda value: tensor.to(copy=value),
            1,
            {"copy": object()},
        )

    def test_torch_function_transient_attribute_error_uses_slot_policy(self):
        tensor = torch.tensor([1.0], dtype=torch.float32)
        descriptor = inspect.getattr_static(torch.Tensor, "to")

        def make_override(attribute_error_at):
            marker = object()
            events = []

            class FlakyDescriptor:
                def __get__(self, obj, objtype=None):
                    events.append("get")
                    if events.count("get") == attribute_error_at:
                        raise AttributeError("transient __torch_function__ miss")

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

            return Override(), marker, events

        for case, call, expected_arg_count, expected_kwargs in (
            ("first positional", lambda value: tensor.to(value), 2, None),
            ("dtype keyword", lambda value: tensor.to(dtype=value), 1, ("dtype",)),
        ):
            value, marker, events = make_override(attribute_error_at=1)
            with self.subTest(case=case):
                self.assertIs(call(value), marker)
                self.assertEqual(
                    events,
                    [
                        "get",
                        "get",
                        "get",
                        (
                            "call",
                            True,
                            ("Override",),
                            expected_arg_count,
                            expected_kwargs,
                        ),
                    ],
                )

        for case, call in (
            ("positional dtype after device", lambda value: tensor.to("cuda", value)),
            ("device keyword", lambda value: tensor.to(device=value)),
            ("copy keyword", lambda value: tensor.to(copy=value)),
            ("non_blocking keyword", lambda value: tensor.to(non_blocking=value)),
            ("memory_format keyword", lambda value: tensor.to(memory_format=value)),
        ):
            value, _, events = make_override(attribute_error_at=1)
            with self.subTest(case=case):
                with self.assertRaises(TypeError):
                    call(value)
                self.assertEqual(events, ["get"])

    def test_tensorbase_descriptor_metadata_documentation_and_unbound_calls(self):
        tensor = torch.tensor([1.0], dtype=torch.float32)
        descriptor = inspect.getattr_static(torch.Tensor, "to")
        bound = tensor.to

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor), "<method 'to' of 'torch._C.TensorBase' objects>"
        )
        self.assertEqual(descriptor.__qualname__, "TensorBase.to")
        self.assertEqual(bound.__qualname__, "Tensor.to")
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        for callable_object in (descriptor, bound):
            self.assertEqual(callable_object.__name__, "to")
            self.assertEqual(callable_object.__doc__, METHOD_DOC)
            self.assertIsNone(callable_object.__text_signature__)
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)

        self.assertIs(descriptor(tensor), tensor)
        self.assertIs(descriptor(tensor, dtype=torch.float32), tensor)
        self.assertIsNot(descriptor(tensor, copy=True), tensor)

    def test_torch_function_modes_receive_original_calls_and_forward(self):
        tensor = torch.tensor([1.0], dtype=torch.float32)
        descriptor = inspect.getattr_static(torch.Tensor, "to")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        cases = (
            ("no args", lambda: tensor.to(), (), None),
            ("cuda", lambda: tensor.to("cuda"), ("cuda",), None),
            (
                "unsupported memory format",
                lambda: tensor.to(memory_format=torch.channels_last),
                (),
                {"memory_format": torch.channels_last},
            ),
        )
        for case, call, positional_tail, keyword_items in cases:
            mode = RecordingMode(marker)
            with mode:
                result = call()
            with self.subTest(case=case):
                self.assertIs(result, marker)
                self.assertEqual(len(mode.calls), 1)
                function, dispatch_types, args, kwargs = mode.calls[0]
                self.assertIs(function, descriptor)
                self.assertEqual(dispatch_types, ())
                self.assertIs(args[0], tensor)
                self.assertEqual(args[1:], positional_tail)
                if keyword_items is None:
                    self.assertIsNone(kwargs)
                else:
                    self.assertEqual(tuple(kwargs), tuple(keyword_items))
                    for key, value in keyword_items.items():
                        self.assertIs(kwargs[key], value)

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append((self.label, func, types, args, kwargs))
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.to()
        self.assertIs(forwarded, tensor)
        self.assertEqual([entry[0] for entry in order], ["upper", "lower"])
        for _, function, dispatch_types, args, kwargs in order:
            self.assertIs(function, descriptor)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(len(args), 1)
            self.assertIs(args[0], tensor)
            self.assertIsNone(kwargs)

        declining = RecordingMode(NotImplemented)
        with self.assertRaisesRegex(
            TypeError,
            r"^Multiple dispatch failed for 'torch\.Tensor\.to'; all "
            r"__torch_function__ handlers returned NotImplemented:",
        ):
            with declining:
                tensor.to()
        self.assertEqual(len(declining.calls), 1)
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

        invalid = RecordingMode(marker)
        with invalid:
            with self.assertRaisesRegex(TypeError, "invalid combination"):
                tensor.to(copy=1)
        self.assertEqual(invalid.calls, [])

    def test_torch_function_overrides_receive_original_calls(self):
        tensor = torch.tensor([1.0], dtype=torch.float32)
        descriptor = inspect.getattr_static(torch.Tensor, "to")
        marker = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        cases = (
            ("first positional", lambda value: tensor.to(value), lambda value: (value,), None),
            (
                "first dtype keyword",
                lambda value: tensor.to(value, dtype=torch.float32),
                lambda value: (value,),
                lambda value: {"dtype": torch.float32},
            ),
            (
                "first dtype positional",
                lambda value: tensor.to(value, torch.float32),
                lambda value: (value, torch.float32),
                None,
            ),
            (
                "first none positional",
                lambda value: tensor.to(value, None),
                lambda value: (value, None),
                None,
            ),
            (
                "first copy keyword",
                lambda value: tensor.to(value, copy=True),
                lambda value: (value,),
                lambda value: {"copy": True},
            ),
            (
                "device dtype positional",
                lambda value: tensor.to("cpu", value),
                lambda value: ("cpu", value),
                None,
            ),
            (
                "cuda dtype positional",
                lambda value: tensor.to("cuda", value),
                lambda value: ("cuda", value),
                None,
            ),
            (
                "dtype keyword",
                lambda value: tensor.to(dtype=value),
                lambda value: (),
                lambda value: {"dtype": value},
            ),
            (
                "device keyword",
                lambda value: tensor.to(device=value),
                lambda value: (),
                lambda value: {"device": value},
            ),
            (
                "nonblocking keyword",
                lambda value: tensor.to(non_blocking=value),
                lambda value: (),
                lambda value: {"non_blocking": value},
            ),
            (
                "copy keyword",
                lambda value: tensor.to(copy=value),
                lambda value: (),
                lambda value: {"copy": value},
            ),
            (
                "memory format keyword",
                lambda value: tensor.to(memory_format=value),
                lambda value: (),
                lambda value: {"memory_format": value},
            ),
        )
        for case, call, expected_tail_factory, expected_kwargs_factory in cases:
            value = Override()
            Override.calls.clear()
            with self.subTest(case=case):
                self.assertIs(call(value), marker)
                self.assertEqual(len(Override.calls), 1)
                function, dispatch_types, args, kwargs = Override.calls[0]
                self.assertIs(function, descriptor)
                self.assertEqual(dispatch_types, (Override,))
                self.assertIs(args[0], tensor)
                expected_tail = expected_tail_factory(value)
                self.assertEqual(len(args), len(expected_tail) + 1)
                for actual, expected in zip(args[1:], expected_tail, strict=True):
                    if expected is value or expected is torch.float32:
                        self.assertIs(actual, expected)
                    else:
                        self.assertEqual(actual, expected)
                expected_kwargs = (
                    None
                    if expected_kwargs_factory is None
                    else expected_kwargs_factory(value)
                )
                if expected_kwargs is None:
                    self.assertIsNone(kwargs)
                else:
                    self.assertEqual(tuple(kwargs), tuple(expected_kwargs))
                    for key, expected in expected_kwargs.items():
                        if expected is value or expected is torch.float32:
                            self.assertIs(kwargs[key], expected)
                        else:
                            self.assertEqual(kwargs[key], expected)

        mode_calls = []

        class DecliningMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                mode_calls.append((func, types, args, kwargs))
                return NotImplemented

        value = Override()
        Override.calls.clear()
        with DecliningMode():
            self.assertIs(tensor.to("cuda", value), marker)
        self.assertEqual(len(mode_calls), 1)
        self.assertEqual(len(Override.calls), 1)
        self.assertIs(mode_calls[0][0], descriptor)
        self.assertEqual(mode_calls[0][1], (Override,))
        self.assertEqual(len(mode_calls[0][2]), 3)
        self.assertIs(mode_calls[0][2][0], tensor)
        self.assertEqual(mode_calls[0][2][1], "cuda")
        self.assertIs(mode_calls[0][2][2], value)
        self.assertIsNone(mode_calls[0][3])
        self.assertIs(Override.calls[0][0], descriptor)
        self.assertEqual(Override.calls[0][1], (Override,))
        self.assertEqual(len(Override.calls[0][2]), 3)
        self.assertIs(Override.calls[0][2][0], tensor)
        self.assertEqual(Override.calls[0][2][1], "cuda")
        self.assertIs(Override.calls[0][2][2], value)
        self.assertIsNone(Override.calls[0][3])


if __name__ == "__main__":
    unittest.main()
