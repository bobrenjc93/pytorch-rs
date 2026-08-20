import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


FUNCTION_NAMES = ("atleast_1d", "atleast_2d", "atleast_3d")


class AtleastZeroInputTests(unittest.TestCase):
    def native_function(self, module, name):
        if module is reference_torch:
            return getattr(module._C._VariableFunctions, name)
        return getattr(module._C, name)

    def zero_input_contract(self, module, name):
        function = getattr(module, name)
        native_function = self.native_function(module, name)
        direct = function()
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        accepting = RecordingMode(marker)
        with accepting:
            accepting_result = function()

        calls = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                calls.append((self.label, func, types, args, kwargs))
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = function()

        three_argument_calls = []

        class ThreeArgumentMode(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=()):
                three_argument_calls.append((func, types, args))
                return marker

        with ThreeArgumentMode():
            three_argument_result = function()

        def observe(call):
            func, types, args, kwargs = call
            return (
                func is native_function,
                func is function,
                types,
                args,
                kwargs,
            )

        def observe_three_argument(call):
            func, types, args = call
            return func is native_function, func is function, types, args

        return (
            (type(direct) is tuple, direct),
            (
                accepting_result is marker,
                tuple(observe(call) for call in accepting.calls),
            ),
            tuple(
                (label, *observe((func, types, args, kwargs)))
                for label, func, types, args, kwargs in calls
            ),
            (type(forwarded) is tuple, forwarded),
            (
                three_argument_result is marker,
                tuple(
                    observe_three_argument(call)
                    for call in three_argument_calls
                ),
            ),
            len(module.overrides._get_current_function_mode_stack()),
        )

    def test_zero_input_results_and_mode_dispatch(self):
        for name in FUNCTION_NAMES:
            with self.subTest(function=name):
                initial_call = (True, False, (), ((),), None)
                forwarded_call = (True, False, (), ((),), {})
                three_argument_call = (True, False, (), ((),))
                expected = (
                    (True, ()),
                    (True, (initial_call,)),
                    (
                        ("upper", *initial_call),
                        ("lower", *forwarded_call),
                    ),
                    (True, ()),
                    (True, (three_argument_call,)),
                    0,
                )
                self.assertEqual(self.zero_input_contract(torch, name), expected)

    @unittest.skipIf(
        reference_torch is None, "install the reference dependency group"
    )
    def test_zero_input_contract_matches_pytorch_2_13(self):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "atleast zero-input differentials require pinned PyTorch 2.13.0"
            )
        for name in FUNCTION_NAMES:
            with self.subTest(function=name):
                self.assertEqual(
                    self.zero_input_contract(torch, name),
                    self.zero_input_contract(reference_torch, name),
                )


if __name__ == "__main__":
    unittest.main()
