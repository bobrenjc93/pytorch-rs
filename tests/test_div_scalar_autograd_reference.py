"""First-order CPU scalar division differentials against PyTorch 2.13."""

import unittest

import numpy as np
import torch_rs as torch
from torch_rs.torch_rs import _nn_functional_dropout_tensor_autograd_suffix

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


FORMS = (
    ("operator", lambda api, x, divisor: x / divisor),
    ("div method", lambda api, x, divisor: x.div(divisor)),
    ("divide method", lambda api, x, divisor: x.divide(divisor)),
    ("div method keywords", lambda api, x, divisor: x.div(other=divisor, rounding_mode=None)),
    ("divide method x2", lambda api, x, divisor: x.divide(x2=divisor, rounding_mode=None)),
    ("div function", lambda api, x, divisor: api.div(x, divisor)),
    ("divide function", lambda api, x, divisor: api.divide(x, divisor)),
    ("div function keywords", lambda api, x, divisor: api.div(input=x, other=divisor, rounding_mode=None, out=None)),
    ("divide function keywords", lambda api, x, divisor: api.divide(input=x, other=divisor, rounding_mode=None)),
)


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class DivScalarAutogradReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(reference_torch.__version__)

    def assert_tensor_matches(self, actual, expected, *, mean_forward=False):
        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.storage_offset(), expected.storage_offset())
        self.assertEqual(actual.requires_grad, expected.requires_grad)
        self.assertEqual(actual.is_leaf, expected.is_leaf)
        self.assertEqual(str(actual.dtype), str(expected.dtype))
        self.assertEqual(str(actual.device), str(expected.device))
        a = np.asarray(actual.detach()).reshape(-1)
        b = expected.detach().numpy().reshape(-1)
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
        valid = ~np.isnan(b)
        if mean_forward:
            # Full-tensor mean keeps its existing reduction-order allowance;
            # division values and every backward comparison stay bit-exact.
            finite_nonzero = np.isfinite(b) & (b != 0)
            np.testing.assert_array_max_ulp(a[finite_nonzero], b[finite_nonzero], maxulp=1)
            valid &= ~finite_nonzero
        # Exact finite/infinity/zero bits catch division order and signed zero.
        np.testing.assert_array_equal(a.view(np.uint32)[valid], b.view(np.uint32)[valid])

    @staticmethod
    def make_input(api, layout):
        if layout == "scalar":
            leaf = api.tensor(7.0, requires_grad=True)
        elif layout == "empty":
            leaf = api.zeros((2, 0, 3), requires_grad=True)
        else:
            leaf = api.tensor(
                [[[1.0, -2.0], [3.0, -4.0]], [[5.0, -6.0], [7.0, -8.0]]],
                requires_grad=True,
            )
        if layout == "transposed":
            return leaf, leaf.transpose(0, 2)
        if layout == "offset":
            return leaf, leaf[1]
        if layout == "offset transposed":
            return leaf, leaf[1].transpose(0, 1)
        if layout == "transposed leaf":
            leaf = leaf.detach().transpose(0, 2).requires_grad_()
        if layout == "empty":
            return leaf, leaf.transpose(0, 2)
        return leaf, leaf

    def test_weighted_layouts_and_public_forms(self):
        for name, divide in FORMS:
            for layout in ("scalar", "empty", "contiguous", "transposed", "offset", "offset transposed", "transposed leaf"):
                with self.subTest(form=name, layout=layout):
                    actual_leaf, actual_input = self.make_input(torch, layout)
                    expected_leaf, expected_input = self.make_input(reference_torch, layout)
                    actual = divide(torch, actual_input, -3.0)
                    expected = divide(reference_torch, expected_input, -3.0)
                    self.assert_tensor_matches(actual, expected)
                    # PyTorch selects DivBackward2 for the explicit rounding_mode
                    # overload, even when None. Both use the same first-order VJP;
                    # the shared native primitive identifies itself as DivBackward0.
                    expected_node = "DivBackward2" if "keywords" in name or "x2" in name else "DivBackward0"
                    self.assertEqual(type(expected.grad_fn).__name__, expected_node)
                    self.assertEqual(
                        _nn_functional_dropout_tensor_autograd_suffix(actual),
                        ", grad_fn=<DivBackward0>",
                    )
                    if actual.numel():
                        self.assertNotEqual(actual.data_ptr(), actual_input.data_ptr())
                    weights = np.arange(actual.numel(), dtype=np.float32).reshape(tuple(actual.shape)) * 1.25 - 2.0
                    actual_weights = torch.tensor(weights.reshape(-1).tolist()).reshape(tuple(actual.shape))
                    expected_weights = reference_torch.tensor(weights.reshape(-1).tolist()).reshape(tuple(expected.shape))
                    actual_loss = (actual * actual_weights).sum()
                    expected_loss = (expected * expected_weights).sum()
                    actual_loss.backward()
                    expected_loss.backward()
                    self.assert_tensor_matches(actual_leaf.grad, expected_leaf.grad)
                    self.assert_tensor_matches(actual_input, expected_input)
                    for loss in (actual_loss, expected_loss):
                        with self.assertRaisesRegex(RuntimeError, "backward through the graph a second time"):
                            loss.backward()
                    self.assert_tensor_matches(actual_leaf.grad, expected_leaf.grad)

    def test_scalar_conversion_and_float32_extremes(self):
        weights = np.asarray([
            0x00000000, 0x80000000, 0x3f800000, 0xbf800000,
            0x40e00000, 0x7f7fffff, 0x00800000, 0x000116c2,
            0x00000001, 0x80000001, 0x7f800000, 0xff800000, 0x7fc12345,
        ], dtype=np.uint32).view(np.float32)
        divisors = (
            True, False, -2, 3, 16777217, np.bool_(True), np.int64(3),
            np.float32(-3.0), 3.0000001, 0.0, -0.0,
            float("inf"), float("-inf"), float("nan"), 1e40, 1e-50,
            float(np.finfo(np.float32).max), float(np.finfo(np.float32).tiny),
            1e-40, float(np.nextafter(np.float32(0), np.float32(1))),
        )
        for name, divide in FORMS:
            for divisor in divisors:
                with self.subTest(form=name, divisor=divisor, scalar_type=type(divisor).__name__):
                    # Nonfinite inputs must not contaminate the input-independent VJP.
                    actual_leaf = torch.tensor(weights.tolist(), requires_grad=True)
                    expected_leaf = reference_torch.tensor(weights.tolist(), requires_grad=True)
                    actual = divide(torch, actual_leaf, divisor)
                    expected = divide(reference_torch, expected_leaf, divisor)
                    self.assert_tensor_matches(actual, expected)
                    (actual * torch.tensor(weights.tolist())).sum().backward()
                    (expected * reference_torch.tensor(weights.tolist())).sum().backward()
                    self.assert_tensor_matches(actual_leaf.grad, expected_leaf.grad)

    def test_repeated_use_accumulation_and_graph_release(self):
        for name, divide in FORMS:
            with self.subTest(form=name):
                leaves = []
                for api in (torch, reference_torch):
                    leaf = api.tensor([2.0, 4.0], requires_grad=True)
                    output = divide(api, leaf * 2.0, 3.0)
                    loss = (output * api.tensor([7.0, -2.0]) + output).sum()
                    loss.backward()
                    gradient = leaf.grad
                    # A fresh graph accumulates in the same leaf-gradient object.
                    divide(api, leaf, -4.0).sum().backward()
                    self.assertIs(leaf.grad, gradient)
                    self.assertFalse(gradient.requires_grad)
                    with self.assertRaisesRegex(RuntimeError, "backward through the graph a second time"):
                        output.sum().backward()
                    leaves.append(leaf)
                self.assert_tensor_matches(leaves[0].grad, leaves[1].grad)

    def test_no_grad_and_detached_inputs(self):
        for name, divide in FORMS:
            with self.subTest(form=name):
                results = []
                for api in (torch, reference_torch):
                    leaf, input_tensor = self.make_input(api, "offset transposed")
                    with api.no_grad():
                        output = divide(api, input_tensor, -3.0)
                    if api is reference_torch:
                        self.assertIsNone(output.grad_fn)
                    else:
                        self.assertEqual(_nn_functional_dropout_tensor_autograd_suffix(output), "")
                    self.assertFalse(output.requires_grad)
                    self.assertIsNone(leaf.grad)
                    detached_output = divide(api, input_tensor.detach(), 3.0)
                    self.assertFalse(detached_output.requires_grad)
                    results.append((output, detached_output))
                for actual, expected in zip(*results):
                    self.assert_tensor_matches(actual, expected)

    def test_division_composes_with_mean_l1_layouts_and_accumulation(self):
        for layout in ("scalar", "empty", "contiguous", "transposed", "offset", "offset transposed", "transposed leaf"):
            for flags in ((True, False), (False, True), (True, True)):
                for reduction in ({}, {"reduction": "mean"}):
                    with self.subTest(layout=layout, flags=flags, reduction=reduction):
                        results = []
                        for api in (torch, reference_torch):
                            left, x = self.make_input(api, layout)
                            right, y = self.make_input(api, layout)
                            left.requires_grad_(flags[0])
                            right.requires_grad_(flags[1])
                            # Recreate views after changing the leaf recording flags.
                            if not flags[0]:
                                x = x.detach()
                            if not flags[1]:
                                y = y.detach()
                            x, y = api.div(x, -3.0), api.divide(y, 2.0)
                            mean = api.nn.functional.l1_loss(x, y, **reduction)
                            loss = mean / -3.0 + mean / 2.0
                            loss.backward()
                            first = tuple(leaf.grad.clone() if flag else None for leaf, flag in zip((left, right), flags))
                            for old in (loss, mean, x if flags[0] else y):
                                with self.assertRaisesRegex(RuntimeError, "backward through the graph a second time"):
                                    old.sum().backward()
                            for leaf, flag in zip((left, right), flags):
                                if flag:
                                    gradient = leaf.grad
                                    (leaf / 4.0).mean().backward()
                                    self.assertIs(leaf.grad, gradient)
                                else:
                                    self.assertIsNone(leaf.grad)
                            results.append((mean, first, (left.grad, right.grad)))
                        self.assert_tensor_matches(results[0][0], results[1][0], mean_forward=True)
                        for actuals, expecteds in zip(results[0][1:], results[1][1:]):
                            for actual, expected, flag in zip(actuals, expecteds, flags):
                                if flag:
                                    self.assert_tensor_matches(actual, expected)

    def test_shared_division_l1_edges_and_ieee_loss_weights(self):
        for shared in ("same operand", "shared nonleaf", "overlapping views"):
            for divisor in (-3.0, 0.0, -0.0, float("inf"), float("-inf"), float("nan"), 1e-40):
                with self.subTest(shared=shared, divisor=divisor):
                    results = []
                    for api in (torch, reference_torch):
                        leaf = api.tensor([-0.0, 0.0, -2.0, 4.0], requires_grad=True)
                        divided = leaf / 2.0
                        if shared == "same operand":
                            x, y = divided, divided
                        elif shared == "shared nonleaf":
                            x, y = divided, divided / -2.0
                        else:
                            x, y = divided[:3], divided[1:]
                        mean = api.nn.functional.l1_loss(x, y)
                        loss = mean / divisor
                        loss.backward()
                        results.append((loss, leaf.grad))
                    for actual, expected in zip(*results):
                        self.assert_tensor_matches(actual, expected)

    def test_division_does_not_bypass_l1_nonfinite_boundary(self):
        for divisor in (0.0, -0.0, float("nan"), 1e-40):
            with self.subTest(divisor=divisor):
                leaf = torch.tensor([0.0, 4.0], requires_grad=True)
                divided = leaf / divisor
                target = torch.zeros(2, requires_grad=True)
                for reduction in ({}, {"reduction": "mean"}):
                    with self.assertRaises((RuntimeError, NotImplementedError)):
                        torch.nn.functional.l1_loss(divided, target, **reduction)
                    self.assertIsNone(leaf.grad)
                    self.assertIsNone(target.grad)
                # Rejecting the loss must leave the division graph usable.
                divided.sum().backward()
                expected = reference_torch.tensor([0.0, 4.0], requires_grad=True)
                (expected / divisor).sum().backward()
                self.assert_tensor_matches(leaf.grad, expected.grad)

    def test_cuda_division_remains_unsupported(self):
        if not torch.cuda.is_available() or not reference_torch.cuda.is_available():
            self.skipTest("CUDA division boundary requires an NVIDIA GPU")
        leaf = torch.zeros((2, 2), device="cuda:0")
        for name, divide in FORMS:
            with self.subTest(form=name):
                with self.assertRaises((RuntimeError, NotImplementedError)):
                    divide(torch, leaf, 3.0)
        self.assertEqual(leaf.cpu().tolist(), [[0.0, 0.0], [0.0, 0.0]])

    def test_unsupported_forms_leave_graph_and_gradients_unchanged(self):
        leaf = torch.tensor([2.0], requires_grad=True)
        output = leaf / 3.0
        for call in (
            lambda: leaf / torch.tensor(2.0),
            lambda: torch.tensor(2.0) / leaf,
            lambda: 2.0 / leaf,
            lambda: torch.div(2.0, leaf),
            lambda: torch.divide(leaf, torch.tensor(2.0)),
            lambda: leaf.div(2.0, rounding_mode="floor"),
            lambda: torch.divide(leaf, 2.0, rounding_mode="trunc"),
            lambda: output.backward(create_graph=True),
            lambda: output.backward(retain_graph=True),
        ):
            with self.subTest(call=call):
                with self.assertRaises((RuntimeError, NotImplementedError)):
                    call()
                self.assertIsNone(leaf.grad)
        output.backward()
        self.assertEqual(leaf.grad.item(), np.float32(1.0 / 3.0))


if __name__ == "__main__":
    unittest.main()
