import re
import unittest

import torch_rs as torch
import torch_rs.torch_rs as actual_variable_functions

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


FUNCTION_NAMES = ("atleast_1d", "atleast_2d", "atleast_3d")


class AtleastZeroInputTests(unittest.TestCase):
    def zero_input_contract(self, module, variable_functions, name):
        function = getattr(module, name)
        native_function = getattr(variable_functions, name)
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

        class ThreeArgumentMode(module.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=()):
                self.calls.append((func, types, args))
                return self.result

        three_argument = ThreeArgumentMode(marker)
        with three_argument:
            three_argument_result = function()

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

        declining = ThreeArgumentMode(NotImplemented)
        try:
            with declining:
                function()
        except Exception as error:
            declining_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x<address>", str(error)),
            )
        else:
            self.fail(f"{module.__name__}.{name} accepted a declining mode")

        def normalize_call(call):
            func, types, args, kwargs = call
            return func is native_function, types, args, kwargs

        def normalize_three_argument_call(call):
            func, types, args = call
            return func is native_function, types, args

        return {
            "direct": (type(direct) is tuple, direct),
            "accepting": (
                accepting_result is marker,
                tuple(normalize_call(call) for call in accepting.calls),
            ),
            "three_argument": (
                three_argument_result is marker,
                tuple(
                    normalize_three_argument_call(call)
                    for call in three_argument.calls
                ),
            ),
            "forwarding": tuple(
                (label, *normalize_call((func, types, args, kwargs)))
                for label, func, types, args, kwargs in calls
            ),
            "forwarded": (type(forwarded) is tuple, forwarded),
            "declining_error": declining_error,
            "declining_calls": tuple(
                normalize_three_argument_call(call)
                for call in declining.calls
            ),
            "stack_depth": len(
                module.overrides._get_current_function_mode_stack()
            ),
        }

    def test_zero_input_results_and_mode_dispatch(self):
        for name in FUNCTION_NAMES:
            with self.subTest(function=name):
                outcome = self.zero_input_contract(
                    torch, actual_variable_functions, name
                )
                self.assertEqual(outcome["direct"], (True, ()))
                self.assertEqual(
                    outcome["accepting"],
                    (True, ((True, (), ((),), None),)),
                )
                self.assertEqual(
                    outcome["three_argument"],
                    (True, ((True, (), ((),)),)),
                )
                self.assertEqual(
                    outcome["forwarding"],
                    (
                        ("upper", True, (), ((),), None),
                        ("lower", True, (), ((),), {}),
                    ),
                )
                self.assertEqual(outcome["forwarded"], (True, ()))
                self.assertEqual(
                    outcome["declining_calls"],
                    ((True, (), ((),)),),
                )
                self.assertEqual(outcome["declining_error"][0], "TypeError")
                self.assertIn(
                    f"Multiple dispatch failed for 'torch.{name}'",
                    outcome["declining_error"][1],
                )
                self.assertEqual(outcome["stack_depth"], 0)

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
                    self.zero_input_contract(
                        torch, actual_variable_functions, name
                    ),
                    self.zero_input_contract(
                        reference_torch, reference_torch._VF, name
                    ),
                )


if __name__ == "__main__":
    unittest.main()
