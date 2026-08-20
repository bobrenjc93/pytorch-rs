import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


FUNCTION_NAMES = ("atleast_1d", "atleast_2d", "atleast_3d")


class AtleastZeroInputTests(unittest.TestCase):
    def zero_input_contract(self, module, name):
        function = getattr(module, name)
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

        def normalize(call):
            func, types, args, kwargs = call
            # PyTorch enters through its native callable with kwargs=None,
            # while forwarding supplies an explicit empty dictionary.
            return func.__name__, types, args, kwargs or {}

        return (
            (type(direct) is tuple, direct),
            (
                accepting_result is marker,
                tuple(normalize(call) for call in accepting.calls),
            ),
            tuple(
                (label, *normalize((func, types, args, kwargs)))
                for label, func, types, args, kwargs in calls
            ),
            (type(forwarded) is tuple, forwarded),
        )

    def test_zero_input_results_and_mode_dispatch(self):
        for name in FUNCTION_NAMES:
            with self.subTest(function=name):
                expected_call = (name, (), ((),), {})
                expected = (
                    (True, ()),
                    (True, (expected_call,)),
                    (
                        ("upper", *expected_call),
                        ("lower", *expected_call),
                    ),
                    (True, ()),
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
