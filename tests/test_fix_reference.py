import copy
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


SPECIAL_INPUT_BITS = np.asarray(
    (
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x8000_0001,
        0x007F_FFFF,
        0x807F_FFFF,
        0x0080_0000,
        0x8080_0000,
        0x3EFF_FFFF,
        0x3F00_0000,
        0x3F7F_FFFF,
        0x3F80_0000,
        0xBF00_0000,
        0xBF7F_FFFF,
        0xBF80_0000,
        0xBFC0_0000,
        0x3FC0_0000,
        0x7F7F_FFFF,
        0xFF7F_FFFF,
        0x7F80_0000,
        0xFF80_0000,
        0x7F81_2345,
        0xFF81_2345,
        0x7FC1_2345,
        0xFFC5_4321,
    ),
    dtype=np.uint32,
)


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TopLevelFixReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        version = reference_torch.__version__.split("+")[0]
        if version != "2.13.0":
            raise AssertionError(
                "torch.fix differentials require pinned PyTorch 2.13.0"
            )

    @staticmethod
    def make_cases(module):
        base = module.tensor(
            np.linspace(-3.75, 3.75, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist(),
            dtype=module.float32,
        )
        strided = base.transpose(0, 2)
        channels_last = module.tensor(
            np.linspace(-15.0, 15.0, 120, dtype=np.float32)
            .reshape(2, 3, 4, 5)
            .tolist(),
            dtype=module.float32,
        ).contiguous(memory_format=module.channels_last)
        channels_last_3d = module.tensor(
            np.linspace(-90.0, 90.0, 720, dtype=np.float32)
            .reshape(2, 3, 4, 5, 6)
            .tolist(),
            dtype=module.float32,
        ).contiguous(memory_format=module.channels_last_3d)
        return (
            ("scalar", module.tensor(-0.0, dtype=module.float32)),
            (
                "empty offset",
                module.zeros((2, 0, 3), dtype=module.float32)
                .transpose(0, 2)[1],
            ),
            ("empty singleton trailing", module.zeros((0, 1))),
            ("empty singleton middle", module.zeros((0, 1, 2))),
            ("empty singleton surrounding", module.zeros((1, 0, 1))),
            ("offset", strided[1]),
            ("noncontiguous", strided),
            ("channels last", channels_last),
            ("channels last 3d", channels_last_3d),
            (
                "numerical edges",
                module.tensor(memoryview(SPECIAL_INPUT_BITS.view(np.float32))),
            ),
        )

    @staticmethod
    def call_fix(module, tensor, form):
        if form == "positional":
            return module.fix(tensor)
        if form == "out none":
            return module.fix(tensor, out=None)
        if form == "alias and out none":
            return module.fix(x=tensor, out=None)
        return module.fix(**{form: tensor})

    def assert_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual, dtype=np.float32).reshape(-1).view(np.uint32),
                expected.detach().cpu().numpy().reshape(-1).view(np.uint32),
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def test_values_ieee_bits_layouts_and_fresh_storage_match_pytorch_2_13(self):
        forms = (
            "positional",
            "input",
            "x",
            "a",
            "x1",
            "out none",
            "alias and out none",
        )
        for actual_case, expected_case in zip(
            self.make_cases(torch), self.make_cases(reference_torch), strict=True
        ):
            case, actual_input = actual_case
            expected_name, expected_input = expected_case
            self.assertEqual(case, expected_name)
            for form in forms:
                actual = self.call_fix(torch, actual_input, form)
                expected = self.call_fix(reference_torch, expected_input, form)
                self.assert_matches(actual, expected, case=(case, form))
                self.assertFalse(actual.is_set_to(actual_input))
                if actual_input.numel():
                    self.assertNotEqual(actual.data_ptr(), actual_input.data_ptr())
                    self.assertNotEqual(
                        expected.data_ptr(), expected_input.data_ptr()
                    )

    def test_detached_and_no_grad_inputs_match_pytorch_2_13(self):
        actual_leaf = torch.tensor(
            [[-2.75, -0.0, 1.25], [2.5, 4.75, 8.0]], requires_grad=True
        )
        expected_leaf = reference_torch.tensor(
            [[-2.75, -0.0, 1.25], [2.5, 4.75, 8.0]], requires_grad=True
        )
        actual_input = actual_leaf.transpose(0, 1)[1]
        expected_input = expected_leaf.transpose(0, 1)[1]

        forms = (
            "positional",
            "input",
            "x",
            "a",
            "x1",
            "out none",
            "alias and out none",
        )
        for form in forms:
            with torch.no_grad():
                actual = self.call_fix(torch, actual_input, form)
            with reference_torch.no_grad():
                expected = self.call_fix(reference_torch, expected_input, form)
            self.assert_matches(actual, expected, case=(form, "no_grad"))

        self.assert_matches(
            torch.fix(actual_input.detach()),
            reference_torch.fix(expected_input.detach()),
            case="detached",
        )
        self.assertIsNone(actual_leaf.grad)
        self.assertIsNone(expected_leaf.grad)

    @staticmethod
    def callable_contract(module):
        function = module.fix
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
            "distinct_from_trunc": function is not module.trunc,
            "name": function.__name__,
            "qualname": function.__qualname__,
            "module": function.__module__,
            "owner_name": owner.__name__,
            "owner_qualname": owner.__qualname__,
            "owner_module": owner.__module__.replace("torch_rs._C", "torch._C"),
            "owner_path_identity": owner is module._C._VariableFunctionsClass,
            "owner_callable_identity": owner.fix is function,
            "owner_distinct_from_trunc": owner.fix is not owner.trunc,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("fix"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["fix"] is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_exports_and_alias_identity_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )

    @staticmethod
    def dispatch_observation(module):
        tensor = module.tensor([1.25])
        destination = module.tensor([0.0])
        function = module.fix
        marker = object()
        mode_observations = []

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        for call in (
            lambda: function(tensor),
            lambda: function(input=tensor),
            lambda: function(x=tensor),
            lambda: function(tensor, out=None),
            lambda: function(input=tensor, out=None),
            lambda: function(tensor, out=destination),
        ):
            mode = RecordingMode()
            with mode:
                result = call()
            func, dispatch_types, args, kwargs = mode.calls[0]
            mode_observations.append(
                (
                    result is marker,
                    func is function,
                    dispatch_types == (),
                    len(args),
                    kwargs is None,
                    None if kwargs is None else tuple(kwargs),
                )
            )

        override_observations = []

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        for call, keyword in (
            (lambda value: function(value), None),
            (lambda value: function(input=value), "input"),
            (lambda value: function(tensor, out=value), "out"),
            (lambda value: function(x=value, out=None), "x"),
        ):
            value = Override()
            Override.calls.clear()
            result = call(value)
            func, dispatch_types, args, kwargs = Override.calls[0]
            override_observations.append(
                (
                    result is marker,
                    func is function,
                    tuple(item.__name__ for item in dispatch_types),
                    len(args),
                    kwargs is None,
                    None if kwargs is None else tuple(kwargs),
                    keyword is not None
                    and kwargs is not None
                    and kwargs[keyword] is value,
                )
            )

        subclass_order = []

        class BaseOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_order.append(
                    ("base", tuple(item.__name__ for item in types))
                )
                return marker

        class DerivedOverride(BaseOverride):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_order.append(
                    ("derived", tuple(item.__name__ for item in types))
                )
                return marker

        subclass_result = function(BaseOverride(), out=DerivedOverride())

        forward_order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                forward_order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = function(input=tensor, out=None)
        forwarded_observation = (
            forwarded.requires_grad,
            forwarded.is_leaf,
            tuple(forwarded.shape),
            forwarded.stride(),
            forwarded.storage_offset(),
            tuple(np.asarray(forwarded).reshape(-1)),
        )

        fallback_events = []

        class FallbackOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                fallback_events.append("override")
                return marker

        declining_mode = RecordingMode(NotImplemented)
        with declining_mode:
            fallback_result = function(FallbackOverride())

        invalid_observations = []
        for call in (
            lambda: function(),
            lambda: function([], out=destination),
            lambda: function(tensor, out=[]),
            lambda: function(tensor, extra=True),
            lambda: function(tensor, tensor),
        ):
            mode = RecordingMode()
            try:
                with mode:
                    call()
            except Exception as error:
                invalid_observations.append(
                    (type(error).__name__, str(error), len(mode.calls))
                )

        return (
            mode_observations,
            override_observations,
            subclass_result is marker,
            subclass_order,
            forward_order,
            forwarded_observation,
            fallback_result is marker,
            len(declining_mode.calls),
            fallback_events,
            invalid_observations,
        )

    def test_mode_and_subclass_override_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.dispatch_observation(torch),
            self.dispatch_observation(reference_torch),
        )

    def test_declining_override_diagnostics_match_pytorch_2_13(self):
        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        self.assert_error_matches(
            lambda: torch.fix(Override()),
            lambda: reference_torch.fix(Override()),
        )
        self.assert_error_matches(
            lambda: torch.fix(torch.tensor([1.0]), out=Override()),
            lambda: reference_torch.fix(
                reference_torch.tensor([1.0]), out=Override()
            ),
        )

    def test_binding_and_type_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.25])
        expected = reference_torch.tensor([1.25])
        cases = (
            (lambda: torch.fix(), lambda: reference_torch.fix()),
            (
                lambda: torch.fix(actual, actual),
                lambda: reference_torch.fix(expected, expected),
            ),
            (
                lambda: torch.fix(actual, input=actual),
                lambda: reference_torch.fix(expected, input=expected),
            ),
            (
                lambda: torch.fix(out=actual),
                lambda: reference_torch.fix(out=expected),
            ),
            (
                lambda: torch.fix(extra=actual),
                lambda: reference_torch.fix(extra=expected),
            ),
            (
                lambda: torch.fix(1, extra=True),
                lambda: reference_torch.fix(1, extra=True),
            ),
            (
                lambda: torch.fix(input=[]),
                lambda: reference_torch.fix(input=[]),
            ),
            (
                lambda: torch.fix(actual, out=[]),
                lambda: reference_torch.fix(expected, out=[]),
            ),
            (
                lambda: torch.fix(actual, extra=True, out=[]),
                lambda: reference_torch.fix(expected, extra=True, out=[]),
            ),
            (
                lambda: torch.fix(actual, extra=True),
                lambda: reference_torch.fix(expected, extra=True),
            ),
            (
                lambda: torch.fix(input=actual, a=actual),
                lambda: reference_torch.fix(input=expected, a=expected),
            ),
            (
                lambda: torch.fix(a=actual, x=actual, out=None),
                lambda: reference_torch.fix(a=expected, x=expected, out=None),
            ),
            (
                lambda: torch.fix(x=actual, a=actual, out=None),
                lambda: reference_torch.fix(x=expected, a=expected, out=None),
            ),
            (
                lambda: torch.fix(np.zeros((2, 3), dtype=np.float32)),
                lambda: reference_torch.fix(
                    np.zeros((2, 3), dtype=np.float32)
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_deliberately_unsupported_surface_remains_narrow(self):
        actual = torch.tensor([1.25], requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError, r"^fix\(\): autograd recording is not supported$"
        ):
            torch.fix(actual)
        expected = reference_torch.tensor([1.25], requires_grad=True)
        self.assertTrue(reference_torch.fix(expected).requires_grad)

        destination = torch.tensor([17.0])
        with self.assertRaisesRegex(
            RuntimeError, r"^fix\(\): the 'out' argument is not supported$"
        ):
            torch.fix(torch.tensor([1.25]), out=destination)
        self.assertEqual(destination.tolist(), [17.0])

        reference_destination = reference_torch.tensor([17.0])
        self.assertIs(
            reference_torch.fix(
                reference_torch.tensor([1.25]), out=reference_destination
            ),
            reference_destination,
        )
        self.assertEqual(reference_destination.tolist(), [1.0])

        self.assertFalse(hasattr(torch.Tensor, "fix"))
        self.assertFalse(hasattr(torch.Tensor, "fix_"))
        self.assertFalse(hasattr(torch, "fix_"))
        self.assertTrue(hasattr(reference_torch.Tensor, "fix"))
        self.assertTrue(hasattr(reference_torch.Tensor, "fix_"))
        self.assertTrue(hasattr(reference_torch, "fix_"))


if __name__ == "__main__":
    unittest.main()
