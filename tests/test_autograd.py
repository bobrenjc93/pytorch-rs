import gc
import inspect
import statistics
import threading
import time
import types
import unittest
import weakref

import numpy as np
import torch_rs as torch

if __package__:
    from .signature_utils import assert_no_argument_signature
else:
    from signature_utils import assert_no_argument_signature

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class AutogradApiTests(unittest.TestCase):
    def test_is_grad_enabled_state_restoration_and_thread_isolation(self):
        self.assertIs(torch.is_grad_enabled(), True)

        with torch.no_grad():
            self.assertIs(torch.is_grad_enabled(), False)
            with torch.no_grad():
                self.assertIs(torch.is_grad_enabled(), False)
            self.assertIs(torch.is_grad_enabled(), False)
        self.assertIs(torch.is_grad_enabled(), True)

        @torch.no_grad()
        def decorated():
            return torch.is_grad_enabled()

        self.assertIs(decorated(), False)
        self.assertIs(torch.is_grad_enabled(), True)

        with self.assertRaisesRegex(RuntimeError, "restore grad mode"):
            with torch.no_grad():
                self.assertIs(torch.is_grad_enabled(), False)
                raise RuntimeError("restore grad mode")
        self.assertIs(torch.is_grad_enabled(), True)

        worker_states = []
        with torch.no_grad():
            thread = threading.Thread(
                target=lambda: worker_states.extend(
                    [torch.is_grad_enabled(), decorated(), torch.is_grad_enabled()]
                )
            )
            thread.start()
            thread.join()
            self.assertIs(torch.is_grad_enabled(), False)
        self.assertEqual(worker_states, [True, False, True])
        self.assertIs(torch.is_grad_enabled(), True)

    def test_is_grad_enabled_public_contract_and_argument_errors(self):
        function = torch.is_grad_enabled
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "is_grad_enabled")
        self.assertEqual(function.__module__, torch.tensor.__module__)
        self.assertEqual(
            function.__doc__,
            "\nis_grad_enabled() -> (bool)\n\n"
            "Returns True if grad mode is currently enabled.\n",
        )
        assert_no_argument_signature(self, function, "()")
        self.assertIn("is_grad_enabled", torch.__all__)

        cases = (
            (
                lambda: function(None),
                "torch.is_grad_enabled() takes no arguments (1 given)",
            ),
            (
                lambda: function(None, None),
                "torch.is_grad_enabled() takes no arguments (2 given)",
            ),
            (
                lambda: function(enabled=True),
                "torch.is_grad_enabled() takes no keyword arguments",
            ),
            (
                lambda: function(None, enabled=True),
                "torch.is_grad_enabled() takes no keyword arguments",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_requires_grad_leaf_grad_accumulation_and_nonleaf_grad(self):
        x = torch.tensor([-2.0, 0.5, 3.0], requires_grad=True)
        self.assertTrue(x.requires_grad)
        self.assertIsNone(x.grad)

        square = x * x
        loss = square.sum()
        self.assertTrue(square.requires_grad)
        self.assertTrue(loss.requires_grad)
        self.assertIsNone(square.grad)
        loss.backward()
        retained_grad = x.grad
        self.assertIs(retained_grad, x.grad)
        np.testing.assert_array_equal(np.asarray(retained_grad), [-4.0, 1.0, 6.0])

        (x * x).sum().backward()
        self.assertIs(retained_grad, x.grad)
        np.testing.assert_array_equal(np.asarray(retained_grad), [-8.0, 2.0, 12.0])

    def test_real_scalar_addition_retains_gradient_history(self):
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        forward = values + 2.0
        reflected = 3.0 + forward
        self.assertTrue(forward.requires_grad)
        self.assertTrue(reflected.requires_grad)

        reflected.sum().backward()
        np.testing.assert_array_equal(np.asarray(values.grad), np.ones((2, 2)))

        repeated_values = torch.tensor([2.0, 3.0], requires_grad=True)
        repeated_loss = (repeated_values + 1.0).sum()
        repeated_loss.backward()
        repeated_loss.backward()
        np.testing.assert_array_equal(np.asarray(repeated_values.grad), [2.0, 2.0])

        detached = values.detach() + 1.0
        self.assertFalse(detached.requires_grad)
        with torch.no_grad():
            self.assertFalse((1.0 + values).requires_grad)

    def test_real_scalar_subtraction_retains_gradient_history(self):
        weights = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        forward_leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        forward = forward_leaf.transpose(0, 1) - 2.0
        self.assertTrue(forward.requires_grad)
        self.assertEqual(forward.stride(), (1, 3))
        np.testing.assert_array_equal(
            np.asarray(forward), [[-1.0, 2.0], [0.0, 3.0], [1.0, 4.0]]
        )
        (forward * weights).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(forward_leaf.grad), [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]
        )

        reflected_leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        reflected = 10.0 - reflected_leaf.transpose(0, 1)
        self.assertTrue(reflected.requires_grad)
        self.assertEqual(reflected.stride(), (1, 3))
        np.testing.assert_array_equal(
            np.asarray(reflected), [[9.0, 6.0], [8.0, 5.0], [7.0, 4.0]]
        )
        (reflected * weights).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(reflected_leaf.grad),
            [[-1.0, -3.0, -5.0], [-2.0, -4.0, -6.0]],
        )

        for operation in (
            lambda value: value - 7.0,
            lambda value: 7.0 - value,
        ):
            empty_leaf = torch.tensor(
                np.empty((0,), dtype=np.float32), requires_grad=True
            )
            empty_output = operation(empty_leaf.reshape(2, 0, 3))
            self.assertTrue(empty_output.requires_grad)
            self.assertEqual(empty_output.stride(), (3, 3, 1))
            empty_output.sum().backward()
            self.assertEqual(empty_leaf.grad.shape, (0,))
            self.assertEqual(empty_leaf.grad.numel(), 0)

        for operation, expected in (
            (lambda value: value - 1.0, [2.0, 2.0]),
            (lambda value: 1.0 - value, [-2.0, -2.0]),
        ):
            repeated_leaf = torch.tensor([2.0, 3.0], requires_grad=True)
            repeated_loss = operation(repeated_leaf).sum()
            repeated_loss.backward()
            repeated_loss.backward()
            np.testing.assert_array_equal(
                np.asarray(repeated_leaf.grad), expected
            )

        self.assertFalse((forward_leaf.detach() - 1.0).requires_grad)
        self.assertFalse((1.0 - forward_leaf.detach()).requires_grad)
        with torch.no_grad():
            self.assertFalse((forward_leaf - 1.0).requires_grad)
            self.assertFalse((1.0 - forward_leaf).requires_grad)

    def test_unary_negation_records_gradients_and_obeys_no_grad(self):
        values = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        weights = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        negated = -values.transpose(0, 1)
        self.assertTrue(negated.requires_grad)
        self.assertEqual(negated.stride(), (1, 3))

        (negated * weights).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(values.grad),
            -np.asarray(weights).transpose(1, 0),
        )

        with torch.no_grad():
            untracked = -values.transpose(0, 1)
            self.assertFalse(untracked.requires_grad)
            self.assertEqual(untracked.stride(), (1, 3))
        self.assertTrue((-values).requires_grad)

    def test_named_neg_matches_operator_autograd_and_no_grad(self):
        method_values = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        operator_values = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        weights = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        method_output = method_values.transpose(0, 1).neg()
        operator_output = -operator_values.transpose(0, 1)
        self.assertEqual(method_output.stride(), operator_output.stride())
        self.assertEqual(method_output.requires_grad, operator_output.requires_grad)
        np.testing.assert_array_equal(
            np.asarray(method_output), np.asarray(operator_output)
        )
        (method_output * weights).sum().backward()
        (operator_output * weights).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(method_values.grad), np.asarray(operator_values.grad)
        )

        method_empty = torch.zeros((2, 0, 3), requires_grad=True)
        operator_empty = torch.zeros((2, 0, 3), requires_grad=True)
        method_empty.neg().sum().backward()
        (-operator_empty).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(method_empty.grad), np.asarray(operator_empty.grad)
        )

        with torch.no_grad():
            method_untracked = method_values.transpose(0, 1).neg()
            operator_untracked = -operator_values.transpose(0, 1)
        self.assertFalse(method_untracked.requires_grad)
        self.assertEqual(method_untracked.requires_grad, operator_untracked.requires_grad)
        self.assertEqual(method_untracked.stride(), operator_untracked.stride())
        np.testing.assert_array_equal(
            np.asarray(method_untracked), np.asarray(operator_untracked)
        )
        self.assertTrue(method_values.neg().requires_grad)

    def test_negative_alias_matches_neg_autograd_reuse_and_no_grad(self):
        def tensor_bits(tensor):
            return (
                np.asarray(tensor, dtype=np.float32)
                .reshape(-1)
                .view(np.uint32)
                .tobytes()
            )

        def snapshot(method_name):
            values = torch.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
            )
            weights = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            output = getattr(values.transpose(0, 1), method_name)()
            output_snapshot = (
                output.shape,
                output.stride(),
                output.storage_offset(),
                output.requires_grad,
                tensor_bits(output),
            )
            (output * weights).sum().backward()
            gradient_snapshot = tensor_bits(values.grad)

            empty = torch.zeros((2, 0, 3), requires_grad=True)
            empty_output = getattr(empty, method_name)()
            empty_output.sum().backward()
            empty_snapshot = (
                empty_output.shape,
                empty_output.stride(),
                empty_output.requires_grad,
                empty.grad.shape,
                empty.grad.stride(),
                tensor_bits(empty.grad),
            )

            repeated = torch.tensor([2.0, 3.0], requires_grad=True)
            repeated_loss = getattr(repeated, method_name)().sum()
            repeated_loss.backward()
            repeated_loss.backward()
            repeated_snapshot = tensor_bits(repeated.grad)

            shared = torch.tensor([5.0, 7.0], requires_grad=True)
            shared_output = getattr(shared, method_name)()
            shared_output.sum().backward()
            shared_output.sum().backward()
            shared_snapshot = tensor_bits(shared.grad)

            nan_bits = np.asarray((0x7FC1_2345, 0xFFC5_4321), dtype=np.uint32)
            nan_weights = torch.tensor(memoryview(nan_bits.view(np.float32)))
            nan_values = torch.tensor([1.0, 2.0], requires_grad=True)
            (getattr(nan_values, method_name)() * nan_weights).sum().backward()
            nan_gradient_snapshot = tensor_bits(nan_values.grad)

            with torch.no_grad():
                untracked = getattr(values.transpose(0, 1), method_name)()
            no_grad_snapshot = (
                untracked.shape,
                untracked.stride(),
                untracked.storage_offset(),
                untracked.requires_grad,
                tensor_bits(untracked),
                getattr(values, method_name)().requires_grad,
            )

            return (
                output_snapshot,
                gradient_snapshot,
                empty_snapshot,
                repeated_snapshot,
                shared_snapshot,
                nan_gradient_snapshot,
                no_grad_snapshot,
            )

        self.assertEqual(snapshot("negative"), snapshot("neg"))

    def test_unary_negation_gradient_is_reusable_shared_and_bitwise(self):
        repeated_values = torch.tensor([2.0, 3.0], requires_grad=True)
        repeated_loss = (-repeated_values).sum()
        repeated_loss.backward()
        repeated_loss.backward()
        np.testing.assert_array_equal(
            np.asarray(repeated_values.grad), [-2.0, -2.0]
        )

        shared_values = torch.tensor([5.0, 7.0], requires_grad=True)
        shared_negative = -shared_values
        first_root = shared_negative.sum()
        second_root = shared_negative.sum()
        first_root.backward()
        second_root.backward()
        np.testing.assert_array_equal(
            np.asarray(shared_values.grad), [-2.0, -2.0]
        )

        nan_bits = np.asarray((0x7FC1_2345, 0xFFC5_4321), dtype=np.uint32)
        weights = torch.tensor(memoryview(nan_bits.view(np.float32)))
        nan_values = torch.tensor([1.0, 2.0], requires_grad=True)
        ((-nan_values) * weights).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(nan_values.grad).view(np.uint32),
            np.asarray((0xFFC1_2345, 0x7FC5_4321), dtype=np.uint32),
        )

    def test_saved_live_gradient_values_are_frozen_at_forward(self):
        source = torch.tensor([4.0, 5.0], requires_grad=True)
        source.sum().backward()
        live_grad = source.grad
        weights = torch.tensor([2.0, 3.0], requires_grad=True)
        saved_loss = (weights * live_grad).sum()

        source.sum().backward()
        np.testing.assert_array_equal(np.asarray(live_grad), [2.0, 2.0])
        saved_loss.backward()
        np.testing.assert_array_equal(np.asarray(weights.grad), [1.0, 1.0])

    def test_requires_grad_requires_a_builtin_bool(self):
        class Truthy:
            def __bool__(self):
                return True

        for invalid in (np.bool_(True), np.bool_(False), 1, 0, None, "true", Truthy()):
            with self.subTest(value=invalid):
                with self.assertRaisesRegex(TypeError, "requires_grad.*must be bool"):
                    torch.tensor([1.0], requires_grad=invalid)

        self.assertTrue(torch.tensor([1.0], requires_grad=True).requires_grad)
        self.assertFalse(torch.tensor([1.0], requires_grad=False).requires_grad)

    def test_zeros_and_ones_create_scalar_ordinary_and_empty_leaves(self):
        weights = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        for name in ("zeros", "ones"):
            factory = getattr(torch, name)
            with self.subTest(factory=name):
                self.assertFalse(factory((), requires_grad=None).requires_grad)
                self.assertFalse(factory((), requires_grad=False).requires_grad)

                scalar = factory((), requires_grad=True)
                self.assertTrue(scalar.requires_grad)
                self.assertIsNone(scalar.grad)
                ((scalar + 2.0) * 3.0).backward()
                self.assertEqual(scalar.grad.shape, ())
                self.assertEqual(scalar.grad.item(), 3.0)

                ordinary = factory((2, 2), requires_grad=True)
                self.assertTrue(ordinary.requires_grad)
                self.assertIsNone(ordinary.grad)
                ((ordinary + 2.0) * weights).sum().backward()
                np.testing.assert_array_equal(
                    np.asarray(ordinary.grad),
                    [[1.0, 2.0], [3.0, 4.0]],
                )

                empty = factory((2, 0, 3), requires_grad=True)
                self.assertTrue(empty.requires_grad)
                self.assertIsNone(empty.grad)
                (empty + 2.0).sum().backward()
                self.assertEqual(empty.grad.shape, (2, 0, 3))
                self.assertEqual(empty.grad.numel(), 0)

    def test_zeros_and_ones_require_keyword_only_builtin_bool_or_none(self):
        class Truthy:
            def __bool__(self):
                return True

        invalid = (
            (np.bool_(True), "numpy.bool"),
            (np.bool_(False), "numpy.bool"),
            (1, "int"),
            (0, "int"),
            (1.0, "float"),
            ("true", "str"),
            (Truthy(), "Truthy"),
            (object(), "object"),
        )
        for name in ("zeros", "ones"):
            factory = getattr(torch, name)
            parameter = inspect.signature(factory).parameters["requires_grad"]
            self.assertIs(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
            self.assertIs(parameter.default, False)

            for value, type_name in invalid:
                with self.subTest(factory=name, value=value):
                    with self.assertRaises(TypeError) as raised:
                        factory((1,), requires_grad=value)
                    self.assertEqual(
                        str(raised.exception),
                        f"{name}(): argument 'requires_grad' must be bool, not {type_name}",
                    )

            with self.subTest(factory=name, argument="positional"):
                with self.assertRaises(TypeError) as raised:
                    factory((1,), True)
                self.assertEqual(
                    str(raised.exception),
                    f"{name}() takes 1 positional argument but 2 were given",
                )

            for competing_keyword in (
                {"wat": 1},
                {"size": (1,)},
                {"requires_grad": 1},
            ):
                with self.subTest(
                    factory=name,
                    argument="positional",
                    competing_keyword=competing_keyword,
                ):
                    with self.assertRaises(TypeError) as raised:
                        factory((1,), True, **competing_keyword)
                    self.assertEqual(
                        str(raised.exception),
                        f"{name}() takes 1 positional argument but 2 were given",
                    )

            mixed_invalid = (
                lambda: factory("bad", requires_grad=1),
                lambda: factory((1,), dtype=object(), requires_grad=1),
                lambda: factory((1,), device=object(), requires_grad=1),
            )
            for call in mixed_invalid:
                with self.subTest(factory=name, argument="mixed invalid"):
                    with self.assertRaises(TypeError) as raised:
                        call()
                    self.assertNotIn("argument 'requires_grad'", str(raised.exception))

            with self.subTest(factory=name, argument="deferred device validation"):
                with self.assertRaises(TypeError) as raised:
                    factory((1,), device="not-a-device", requires_grad=1)
                self.assertEqual(
                    str(raised.exception),
                    f"{name}(): argument 'requires_grad' must be bool, not int",
                )

    def test_eye_creates_square_rectangular_and_empty_leaves(self):
        cases = (
            ((2,), (2, 2), [[1.0, 0.0], [0.0, 1.0]]),
            ((2, 3), (2, 3), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            ((0, 3), (0, 3), []),
            ((3, 0), (3, 0), [[], [], []]),
        )
        for arguments, shape, values in cases:
            with self.subTest(shape=shape):
                for requires_grad in (None, False):
                    ordinary = torch.eye(*arguments, requires_grad=requires_grad)
                    self.assertFalse(ordinary.requires_grad)
                    self.assertEqual(ordinary.tolist(), values)

                omitted = torch.eye(*arguments)
                self.assertFalse(omitted.requires_grad)
                self.assertEqual(omitted.tolist(), values)

                leaf = torch.eye(*arguments, requires_grad=True)
                self.assertTrue(leaf.requires_grad)
                self.assertTrue(leaf.is_leaf)
                self.assertIsNone(leaf.grad)
                self.assertEqual(leaf.tolist(), values)

                weights = torch.ones(shape)
                loss = (leaf * weights).sum()
                self.assertTrue(loss.requires_grad)
                loss.backward()
                self.assertEqual(leaf.grad.shape, shape)
                np.testing.assert_array_equal(
                    np.asarray(leaf.grad),
                    np.ones(shape, dtype=np.float32),
                )

    def test_eye_requires_keyword_only_builtin_bool_or_none(self):
        class Truthy:
            def __bool__(self):
                return True

        with self.assertRaises(ValueError):
            inspect.signature(torch.eye)

        invalid = (
            (np.bool_(True), "numpy.bool"),
            (np.bool_(False), "numpy.bool"),
            (1, "int"),
            (0, "int"),
            (1.0, "float"),
            ("true", "str"),
            (Truthy(), "Truthy"),
            (object(), "object"),
        )
        for value, type_name in invalid:
            with self.subTest(value=value):
                with self.assertRaises(TypeError) as raised:
                    torch.eye(1, requires_grad=value)
                self.assertEqual(
                    str(raised.exception),
                    f"eye(): argument 'requires_grad' must be bool, not {type_name}",
                )

        competing_positionals = (
            lambda: torch.eye(1, 1, True),
            lambda: torch.eye(1, 1, True, requires_grad=1),
            lambda: torch.eye(1, 1, True, wat=1),
            lambda: torch.eye(1, 1, True, n=1),
        )
        for call in competing_positionals:
            with self.subTest(argument="positional"):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(
                    str(raised.exception),
                    "eye() takes from 1 to 2 positional arguments but 3 were given",
                )

        metadata_precedence = (
            lambda: torch.eye(2**63, dtype=object(), requires_grad=1),
            lambda: torch.eye(2**63, device=object(), requires_grad=1),
        )
        for call in metadata_precedence:
            with self.subTest(argument="metadata precedence"):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertNotIn("argument 'requires_grad'", str(raised.exception))

        requires_grad_precedence = (
            lambda: torch.eye(2**63, requires_grad=1),
            lambda: torch.eye(-1, requires_grad=1),
            lambda: torch.eye(1, -1, requires_grad=1),
            lambda: torch.eye(1, device="not-a-device", requires_grad=1),
            lambda: torch.eye(1, wat=1, requires_grad=1),
            lambda: torch.eye(1, n=1, requires_grad=1),
            lambda: torch.eye(1, 1, m=1, requires_grad=1),
        )
        for call in requires_grad_precedence:
            with self.subTest(argument="requires_grad precedence"):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(
                    str(raised.exception),
                    "eye(): argument 'requires_grad' must be bool, not int",
                )

        with self.assertRaises(TypeError) as raised:
            torch.eye(requires_grad=1)
        self.assertEqual(
            str(raised.exception),
            "eye() missing 1 required positional argument: 'n'",
        )

    def test_full_creates_scalar_ordinary_and_empty_leaves(self):
        for requires_grad in (None, False):
            with self.subTest(requires_grad=requires_grad):
                scalar = torch.full((), -2.5, requires_grad=requires_grad)
                self.assertFalse(scalar.requires_grad)
                self.assertEqual(scalar.item(), -2.5)

        omitted = torch.full((2,), 3.25)
        self.assertFalse(omitted.requires_grad)
        self.assertEqual(omitted.tolist(), [3.25, 3.25])

        scalar = torch.full((), -2.5, requires_grad=True)
        self.assertTrue(scalar.requires_grad)
        self.assertIsNone(scalar.grad)
        ((scalar + 2.0) * 3.0).backward()
        self.assertEqual(scalar.grad.shape, ())
        self.assertEqual(scalar.grad.item(), 3.0)

        ordinary = torch.full((2, 2), 3.25, requires_grad=True)
        self.assertTrue(ordinary.requires_grad)
        self.assertIsNone(ordinary.grad)
        weights = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        ((ordinary + 2.0) * weights).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(ordinary.grad),
            [[1.0, 2.0], [3.0, 4.0]],
        )

        empty = torch.full((2, 0, 3), 7.0, requires_grad=True)
        self.assertTrue(empty.requires_grad)
        self.assertIsNone(empty.grad)
        self.assertEqual(empty.tolist(), [[], []])
        (empty + 2.0).sum().backward()
        self.assertEqual(empty.grad.shape, (2, 0, 3))
        self.assertEqual(empty.grad.numel(), 0)

    def test_full_requires_keyword_only_builtin_bool_or_none(self):
        class Truthy:
            def __bool__(self):
                return True

        parameter = inspect.signature(torch.full).parameters["requires_grad"]
        self.assertIs(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIs(parameter.default, False)

        invalid = (
            (np.bool_(True), "numpy.bool"),
            (np.bool_(False), "numpy.bool"),
            (1, "int"),
            (0, "int"),
            (1.0, "float"),
            ("true", "str"),
            (Truthy(), "Truthy"),
            (object(), "object"),
        )
        for value, type_name in invalid:
            with self.subTest(value=value):
                with self.assertRaises(TypeError) as raised:
                    torch.full((1,), 2.0, requires_grad=value)
                self.assertEqual(
                    str(raised.exception),
                    f"full(): argument 'requires_grad' must be bool, not {type_name}",
                )

        for competing_keyword in (
            {"wat": 1},
            {"size": (1,)},
            {"fill_value": 2.0},
            {"requires_grad": 1},
        ):
            with self.subTest(competing_keyword=competing_keyword):
                with self.assertRaises(TypeError) as raised:
                    torch.full((1,), 2.0, True, **competing_keyword)
                self.assertEqual(
                    str(raised.exception),
                    "full() takes 2 positional arguments but 3 were given",
                )

        mixed_invalid = (
            lambda: torch.full("bad", 2.0, requires_grad=1),
            lambda: torch.full((1,), object(), requires_grad=1),
            lambda: torch.full((1,), 2.0, dtype=object(), requires_grad=1),
            lambda: torch.full((1,), 2.0, device=object(), requires_grad=1),
        )
        for call in mixed_invalid:
            with self.subTest(argument="mixed invalid"):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertNotIn("argument 'requires_grad'", str(raised.exception))

        requires_grad_precedence = (
            lambda: torch.full((-1,), 2.0, requires_grad=1),
            lambda: torch.full((1,), 2.0, device="not-a-device", requires_grad=1),
            lambda: torch.full((1,), 2.0, wat=1, requires_grad=1),
            lambda: torch.full((1,), 2.0, size=(1,), requires_grad=1),
            lambda: torch.full((1,), 2.0, fill_value=2.0, requires_grad=1),
        )
        for call in requires_grad_precedence:
            with self.subTest(argument="requires_grad precedence"):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(
                    str(raised.exception),
                    "full(): argument 'requires_grad' must be bool, not int",
                )

    def test_detach_and_no_grad_context_decorator_are_boundaries(self):
        x = torch.tensor([2.0, 3.0], requires_grad=True)
        detached = x.detach()
        self.assertFalse(detached.requires_grad)
        self.assertEqual(detached.tolist(), x.tolist())
        self.assertFalse((detached * detached).sum().requires_grad)

        with torch.no_grad():
            self.assertFalse((x * x).requires_grad)
            with torch.no_grad():
                self.assertFalse(x.sum().requires_grad)
            self.assertFalse((x * 4.0).requires_grad)
        self.assertTrue((x * x).requires_grad)

        @torch.no_grad()
        def square(value):
            return value * value

        self.assertFalse(square(x).requires_grad)
        self.assertTrue((x * x).requires_grad)

        with self.assertRaisesRegex(ValueError, "restore recording"):
            with torch.no_grad():
                raise ValueError("restore recording")
        self.assertTrue((x * x).requires_grad)

    def test_backward_errors_repeated_policy_and_graph_lifetime(self):
        with self.assertRaisesRegex(RuntimeError, "does not require grad"):
            torch.tensor(1.0).backward()
        with self.assertRaisesRegex(RuntimeError, "does not require grad"):
            torch.tensor([1.0, 2.0]).backward()
        with self.assertRaisesRegex(RuntimeError, "implicitly created only for scalar"):
            torch.tensor([1.0, 2.0], requires_grad=True).backward()

        x = torch.tensor([2.0, 3.0], requires_grad=True)
        intermediate = x * x
        output = intermediate.sum()
        del intermediate
        gc.collect()
        output.backward()
        np.testing.assert_array_equal(np.asarray(x.grad), [4.0, 6.0])
        with self.assertRaisesRegex(RuntimeError, "backward through the graph a second time"):
            output.backward()

        scalar_leaf = torch.tensor([7.0], requires_grad=True)
        scalar_leaf.backward()
        scalar_leaf.backward()
        self.assertEqual(scalar_leaf.grad.tolist(), [2.0])

        summed_leaf = torch.tensor([1.0, 2.0], requires_grad=True)
        summed = summed_leaf.sum()
        summed.backward()
        summed.backward()
        self.assertEqual(summed_leaf.grad.tolist(), [2.0, 2.0])

        transformed_leaf = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        transformed = transformed_leaf.T.clone().sum()
        transformed.backward()
        transformed.backward()
        np.testing.assert_array_equal(
            np.asarray(transformed_leaf.grad), [[2.0, 2.0], [2.0, 2.0]]
        )

    def test_deep_graph_transformations_and_negative_zero(self):
        deep_leaf = torch.tensor(3.0, requires_grad=True)
        deep_output = deep_leaf
        for _ in range(20_000):
            deep_output = deep_output * 1.0
        deep_output.backward()
        self.assertEqual(deep_leaf.grad.item(), 1.0)

        leaf = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], requires_grad=True)
        transformed = leaf.transpose(1, 2)
        self.assertTrue(transformed.requires_grad)
        transformed = transformed.contiguous().squeeze(0)[1].reshape(2, 1).clone()
        self.assertTrue(transformed.requires_grad)
        (transformed * transformed).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(leaf.grad),
            np.array([[[0.0, 4.0, 0.0], [0.0, 10.0, 0.0]]], dtype=np.float32),
        )

        matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        self.assertTrue(matrix.T.requires_grad)
        (matrix.T * matrix.T).sum().backward()
        np.testing.assert_array_equal(np.asarray(matrix.grad), [[2.0, 4.0], [6.0, 8.0]])

        signed = torch.tensor([2.0], requires_grad=True)
        (signed * torch.tensor([-0.0])).sum().backward()
        self.assertEqual(np.asarray(signed.grad).view(np.uint32)[0], 0x8000_0000)

        broadcast_signed = torch.tensor(2.0, requires_grad=True)
        (broadcast_signed * torch.tensor([-0.0, -0.0])).sum().backward()
        self.assertEqual(np.asarray(broadcast_signed.grad).view(np.uint32).item(), 0)

    def test_no_grad_decorator_binds_methods_and_is_cross_thread_callable(self):
        class Model:
            @torch.no_grad()
            def forward(self, value, scale=1.0):
                return value * scale

        value = torch.tensor([2.0], requires_grad=True)
        model = Model()
        self.assertFalse(model.forward(value, scale=3.0).requires_grad)
        self.assertFalse(Model.forward(model, value).requires_grad)

        results = []
        failures = []

        @torch.no_grad()
        def worker(input_value):
            return input_value * input_value

        def run_worker():
            try:
                results.append(worker(value).requires_grad)
                results.append((value * value).requires_grad)
            except BaseException as error:  # PanicException inherits BaseException.
                failures.append(error)

        thread = threading.Thread(target=run_worker)
        thread.start()
        thread.join()
        self.assertEqual(failures, [])
        self.assertEqual(results, [False, True])
        self.assertTrue((value * value).requires_grad)

        context = torch.no_grad()
        context_results = []
        context_failures = []

        def run_context():
            try:
                with context:
                    context_results.append((value * value).requires_grad)
                context_results.append((value * value).requires_grad)
            except BaseException as error:
                context_failures.append(error)

        context_thread = threading.Thread(target=run_context)
        context_thread.start()
        context_thread.join()
        self.assertEqual(context_failures, [])
        self.assertEqual(context_results, [False, True])

    def test_no_grad_public_contract_and_callable_cycles(self):
        value = torch.tensor([2.0], requires_grad=True)
        with torch.no_grad() as entered:
            self.assertIsNone(entered)
            self.assertFalse((value * value).requires_grad)

        @torch.no_grad
        def direct(input_value: object, scale: float = 1.0) -> object:
            """A metadata-bearing no-grad callable."""
            return input_value * scale

        self.assertFalse(direct(value, scale=3.0).requires_grad)
        self.assertEqual(direct.__name__, "direct")
        self.assertEqual(direct.__doc__, "A metadata-bearing no-grad callable.")
        self.assertEqual(
            direct.__annotations__,
            {"input_value": object, "scale": float, "return": object},
        )
        self.assertEqual(inspect.signature(direct), inspect.signature(direct.__wrapped__))
        self.assertTrue(gc.is_tracked(direct))

        def make_callable_cycle():
            wrapped = None

            def function():
                return wrapped

            wrapped = torch.no_grad()(function)
            return weakref.ref(wrapped)

        callable_reference = make_callable_cycle()
        gc.collect()
        self.assertIsNone(callable_reference())

    def test_no_grad_decorator_guards_every_generator_resume(self):
        value = torch.tensor([2.0], requires_grad=True)
        events = []

        @torch.no_grad()
        def generate():
            events.append(("next", (value * value).requires_grad))
            request = yield value * value
            events.append(("send", request, (value * value).requires_grad))
            try:
                yield value * value
            except ValueError as error:
                events.append(("throw", str(error), (value * value).requires_grad))
                yield value * value
            finally:
                events.append(("close", (value * value).requires_grad))

        self.assertTrue(inspect.isgeneratorfunction(generate))
        generator = generate()
        self.assertTrue(inspect.isgenerator(generator))
        self.assertTrue(gc.is_tracked(generator))
        self.assertIs(iter(generator), generator)
        self.assertFalse(next(generator).requires_grad)
        self.assertTrue((value * value).requires_grad)
        self.assertFalse(generator.send("request").requires_grad)
        self.assertTrue((value * value).requires_grad)
        self.assertFalse(generator.throw(ValueError("injected")).requires_grad)
        self.assertTrue((value * value).requires_grad)
        self.assertIsNone(generator.close())
        self.assertEqual(
            events,
            [
                ("next", False),
                ("send", "request", False),
                ("throw", "injected", False),
                ("close", False),
            ],
        )
        self.assertTrue((value * value).requires_grad)

        abandoned_events = []

        @torch.no_grad()
        def abandoned():
            try:
                yield value * value
            finally:
                abandoned_events.append((value * value).requires_grad)

        generator = abandoned()
        self.assertFalse(next(generator).requires_grad)
        del generator
        gc.collect()
        self.assertEqual(abandoned_events, [False])
        self.assertTrue((value * value).requires_grad)

        cyclic_events = []

        @torch.no_grad()
        def cyclic():
            proxy = yield None
            try:
                yield proxy
            finally:
                cyclic_events.append((value * value).requires_grad)

        generator = cyclic()
        next(generator)
        generator.send(generator)
        generator_reference = weakref.ref(generator)
        del generator
        gc.collect()
        self.assertIsNone(generator_reference())
        self.assertEqual(cyclic_events, [False])
        self.assertTrue((value * value).requires_grad)

    def test_unconsumed_deep_graph_drop_and_detach_are_stack_safe(self):
        leaf = torch.tensor(3.0, requires_grad=True)
        output = leaf
        for _ in range(100_000):
            output = output * 1.0
        del output
        gc.collect()

        output = leaf
        for _ in range(100_000):
            output = output * 1.0
        detached = output.detach()
        del output
        gc.collect()
        self.assertFalse(detached.requires_grad)
        self.assertEqual(detached.item(), 3.0)


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class AutogradReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("autograd differentials require pinned PyTorch 2.13.0")

    def assert_gradient_pair(self, native, expected):
        self.assertEqual(native.shape, tuple(expected.shape))
        self.assertEqual(native.grad.shape, tuple(expected.grad.shape))
        np.testing.assert_allclose(
            np.asarray(native.grad),
            expected.grad.detach().cpu().numpy(),
            rtol=1.0e-6,
            atol=1.0e-6,
        )

    def test_is_grad_enabled_matches_pytorch_2_13(self):
        outcomes = []
        for module in (torch, reference_torch):
            function = module.is_grad_enabled
            states = [function()]
            with module.no_grad():
                states.append(function())
                with module.no_grad():
                    states.append(function())
                states.append(function())
            states.append(function())

            @module.no_grad()
            def decorated():
                return function()

            states.extend((decorated(), function()))
            try:
                with module.no_grad():
                    states.append(function())
                    raise RuntimeError("restore grad mode")
            except RuntimeError:
                states.append(function())

            worker_states = []
            with module.no_grad():
                thread = threading.Thread(
                    target=lambda: worker_states.extend(
                        [function(), decorated(), function()]
                    )
                )
                thread.start()
                thread.join()
                states.append(function())
            states.append(function())

            errors = []
            for call in (
                lambda: function(None),
                lambda: function(None, None),
                lambda: function(enabled=True),
                lambda: function(None, enabled=True),
            ):
                try:
                    call()
                except TypeError as error:
                    errors.append((type(error).__name__, str(error)))
                else:
                    self.fail(f"{module.__name__}.is_grad_enabled accepted arguments")

            assert_no_argument_signature(self, function, "()")
            outcomes.append(
                (
                    states,
                    worker_states,
                    errors,
                    type(function),
                    function.__name__,
                    function.__text_signature__,
                    function.__doc__,
                    "is_grad_enabled" in module.__all__,
                )
            )

        self.assertEqual(outcomes[0], outcomes[1])

    def test_requires_grad_argument_types_match_pytorch_2_13(self):
        class Truthy:
            def __bool__(self):
                return True

        values = [np.bool_(True), np.bool_(False), 1, 0, None, "true", Truthy()]
        outcomes = []
        for module in (torch, reference_torch):
            errors = []
            for value in values:
                try:
                    module.tensor([1.0], requires_grad=value)
                except TypeError as error:
                    errors.append(str(error))
                else:
                    self.fail(f"{module.__name__} accepted requires_grad={value!r}")
            outcomes.append(errors)

        self.assertEqual(outcomes[0], outcomes[1])

    def test_zeros_and_ones_requires_grad_arguments_match_pytorch_2_13(self):
        class Truthy:
            def __bool__(self):
                return True

        invalid = [
            np.bool_(True),
            np.bool_(False),
            1,
            0,
            1.0,
            "true",
            Truthy(),
            object(),
        ]
        mixed_invalid = (
            lambda factory: factory("bad", requires_grad=1),
            lambda factory: factory((1,), dtype=object(), requires_grad=1),
            lambda factory: factory((1,), device=object(), requires_grad=1),
        )
        competing_errors = (
            lambda factory: factory((1,), True, wat=1),
            lambda factory: factory((1,), True, size=(1,)),
            lambda factory: factory((1,), True, requires_grad=1),
        )
        outcomes = []
        for module in (torch, reference_torch):
            module_outcomes = []
            for name in ("zeros", "ones"):
                factory = getattr(module, name)
                errors = []
                for value in invalid:
                    try:
                        factory((1,), requires_grad=value)
                    except TypeError as error:
                        errors.append(str(error))
                    else:
                        self.fail(
                            f"{module.__name__}.{name} accepted requires_grad={value!r}"
                        )

                try:
                    factory((1,), True)
                except TypeError as error:
                    positional_error = str(error)
                else:
                    self.fail(f"{module.__name__}.{name} accepted positional True")

                mixed_error_precedence = []
                for call in mixed_invalid:
                    try:
                        call(factory)
                    except TypeError as error:
                        mixed_error_precedence.append(
                            (type(error).__name__, "argument 'requires_grad'" in str(error))
                        )
                    else:
                        self.fail(f"{module.__name__}.{name} accepted mixed invalid arguments")

                positional_precedence = []
                for call in competing_errors:
                    try:
                        call(factory)
                    except TypeError as error:
                        positional_precedence.append(str(error))
                    else:
                        self.fail(f"{module.__name__}.{name} accepted excess positionals")

                try:
                    factory((1,), device="not-a-device", requires_grad=1)
                except TypeError as error:
                    deferred_device_error = str(error)
                else:
                    self.fail(f"{module.__name__}.{name} accepted invalid requires_grad")

                module_outcomes.append(
                    (
                        factory((1,)).requires_grad,
                        factory((1,), requires_grad=None).requires_grad,
                        factory((1,), requires_grad=False).requires_grad,
                        factory((1,), requires_grad=True).requires_grad,
                        errors,
                        positional_error,
                        mixed_error_precedence,
                        positional_precedence,
                        deferred_device_error,
                    )
                )
            outcomes.append(module_outcomes)

        self.assertEqual(outcomes[0], outcomes[1])

    def test_zeros_and_ones_leaf_gradients_match_pytorch_2_13(self):
        outcomes = []
        for module in (torch, reference_torch):
            module_outcomes = []
            weights = module.tensor([[1.0, 2.0], [3.0, 4.0]])
            for name in ("zeros", "ones"):
                factory = getattr(module, name)

                scalar = factory((), requires_grad=True)
                ((scalar + 2.0) * 3.0).backward()

                ordinary = factory((2, 2), requires_grad=True)
                ((ordinary + 2.0) * weights).sum().backward()

                empty = factory((2, 0, 3), requires_grad=True)
                (empty + 2.0).sum().backward()

                module_outcomes.append(
                    (
                        scalar.requires_grad,
                        scalar.grad.item(),
                        np.asarray(ordinary.grad).copy(),
                        empty.requires_grad,
                        tuple(empty.grad.shape),
                        empty.grad.numel(),
                    )
                )
            outcomes.append(module_outcomes)

        for native, expected in zip(outcomes[0], outcomes[1]):
            self.assertEqual(native[:2], expected[:2])
            np.testing.assert_array_equal(native[2], expected[2])
            self.assertEqual(native[3:], expected[3:])

    def test_eye_requires_grad_arguments_match_pytorch_2_13(self):
        class Truthy:
            def __bool__(self):
                return True

        invalid = (
            np.bool_(True),
            np.bool_(False),
            1,
            0,
            1.0,
            "true",
            Truthy(),
            object(),
        )
        positional_calls = (
            lambda factory: factory(1, 1, True),
            lambda factory: factory(1, 1, True, requires_grad=1),
            lambda factory: factory(1, 1, True, wat=1),
        )
        precedence_calls = (
            lambda factory: factory(2**63, dtype=object(), requires_grad=1),
            lambda factory: factory(2**63, device=object(), requires_grad=1),
            lambda factory: factory(2**63, requires_grad=1),
            lambda factory: factory(-1, requires_grad=1),
            lambda factory: factory(1, -1, requires_grad=1),
            lambda factory: factory(
                1, device="not-a-device", requires_grad=1
            ),
            lambda factory: factory(1, wat=1, requires_grad=1),
            lambda factory: factory(1, n=1, requires_grad=1),
            lambda factory: factory(requires_grad=1),
        )

        outcomes = []
        for module in (torch, reference_torch):
            factory = module.eye
            invalid_errors = []
            for value in invalid:
                try:
                    factory(1, requires_grad=value)
                except Exception as error:
                    invalid_errors.append(type(error).__name__)
                else:
                    self.fail(
                        f"{module.__name__}.eye accepted requires_grad={value!r}"
                    )

            positional_errors = []
            for call in positional_calls:
                try:
                    call(factory)
                except Exception as error:
                    positional_errors.append(type(error).__name__)
                else:
                    self.fail(f"{module.__name__}.eye accepted positional options")

            precedence_errors = []
            for call in precedence_calls:
                try:
                    call(factory)
                except Exception as error:
                    precedence_errors.append(type(error).__name__)
                else:
                    self.fail(f"{module.__name__}.eye accepted invalid arguments")

            outcomes.append(
                (
                    factory(2).requires_grad,
                    factory(2, requires_grad=None).requires_grad,
                    factory(2, requires_grad=False).requires_grad,
                    factory(2, requires_grad=True).requires_grad,
                    factory(2, requires_grad=True).is_leaf,
                    factory(2, 3, requires_grad=None).tolist(),
                    invalid_errors,
                    positional_errors,
                    precedence_errors,
                )
            )

        self.assertEqual(outcomes[0], outcomes[1])

    def test_eye_leaf_gradients_match_pytorch_2_13(self):
        cases = ((2,), (2, 3), (0, 3), (3, 0))
        outcomes = []
        for module in (torch, reference_torch):
            module_outcomes = []
            for arguments in cases:
                leaf = module.eye(*arguments, requires_grad=True)
                weights = module.ones(tuple(leaf.shape))
                loss = (leaf * weights).sum()
                loss.backward()
                module_outcomes.append(
                    (
                        tuple(leaf.shape),
                        leaf.tolist(),
                        leaf.requires_grad,
                        leaf.is_leaf,
                        np.asarray(leaf.grad).copy(),
                    )
                )
            outcomes.append(module_outcomes)

        for native, expected in zip(outcomes[0], outcomes[1]):
            self.assertEqual(native[:4], expected[:4])
            np.testing.assert_array_equal(native[4], expected[4])

    def test_full_requires_grad_arguments_match_pytorch_2_13(self):
        class Truthy:
            def __bool__(self):
                return True

        invalid = [
            np.bool_(True),
            np.bool_(False),
            1,
            0,
            1.0,
            "true",
            Truthy(),
            object(),
        ]
        mixed_invalid = (
            lambda factory: factory("bad", 2.0, requires_grad=1),
            lambda factory: factory((1,), object(), requires_grad=1),
            lambda factory: factory((1,), 2.0, dtype=object(), requires_grad=1),
            lambda factory: factory((1,), 2.0, device=object(), requires_grad=1),
        )
        requires_grad_precedence = (
            lambda factory: factory((-1,), 2.0, requires_grad=1),
            lambda factory: factory(
                (1,), 2.0, device="not-a-device", requires_grad=1
            ),
            lambda factory: factory((1,), 2.0, wat=1, requires_grad=1),
            lambda factory: factory(
                (1,), 2.0, size=(1,), requires_grad=1
            ),
            lambda factory: factory(
                (1,), 2.0, fill_value=2.0, requires_grad=1
            ),
        )
        competing_errors = (
            lambda factory: factory((1,), 2.0, True, wat=1),
            lambda factory: factory((1,), 2.0, True, size=(1,)),
            lambda factory: factory((1,), 2.0, True, fill_value=2.0),
            lambda factory: factory((1,), 2.0, True, requires_grad=1),
        )

        outcomes = []
        for module in (torch, reference_torch):
            factory = module.full
            errors = []
            for value in invalid:
                try:
                    factory((1,), 2.0, requires_grad=value)
                except TypeError as error:
                    errors.append(str(error))
                else:
                    self.fail(
                        f"{module.__name__}.full accepted requires_grad={value!r}"
                    )

            mixed_error_precedence = []
            for call in mixed_invalid:
                try:
                    call(factory)
                except TypeError as error:
                    mixed_error_precedence.append(
                        (type(error).__name__, "argument 'requires_grad'" in str(error))
                    )
                else:
                    self.fail(f"{module.__name__}.full accepted mixed invalid arguments")

            requires_grad_errors = []
            for call in requires_grad_precedence:
                try:
                    call(factory)
                except TypeError as error:
                    requires_grad_errors.append(str(error))
                else:
                    self.fail(f"{module.__name__}.full accepted invalid requires_grad")

            positional_precedence = []
            for call in competing_errors:
                try:
                    call(factory)
                except TypeError as error:
                    positional_precedence.append(str(error))
                else:
                    self.fail(f"{module.__name__}.full accepted excess positionals")

            outcomes.append(
                (
                    factory((1,), 3.25).requires_grad,
                    factory((1,), 3.25, requires_grad=None).requires_grad,
                    factory((1,), 3.25, requires_grad=False).requires_grad,
                    factory((1,), 3.25, requires_grad=True).requires_grad,
                    factory((2,), 3.25, requires_grad=None).tolist(),
                    errors,
                    mixed_error_precedence,
                    requires_grad_errors,
                    positional_precedence,
                )
            )

        self.assertEqual(outcomes[0], outcomes[1])

    def test_full_leaf_gradients_match_pytorch_2_13(self):
        outcomes = []
        for module in (torch, reference_torch):
            scalar = module.full((), -2.5, requires_grad=True)
            ((scalar + 2.0) * 3.0).backward()

            ordinary = module.full((2, 2), 3.25, requires_grad=True)
            weights = module.tensor([[1.0, 2.0], [3.0, 4.0]])
            ((ordinary + 2.0) * weights).sum().backward()

            empty = module.full((2, 0, 3), 7.0, requires_grad=True)
            (empty + 2.0).sum().backward()

            outcomes.append(
                (
                    scalar.requires_grad,
                    scalar.grad.item(),
                    np.asarray(ordinary.grad).copy(),
                    empty.requires_grad,
                    tuple(empty.grad.shape),
                    empty.grad.numel(),
                )
            )

        self.assertEqual(outcomes[0][:2], outcomes[1][:2])
        np.testing.assert_array_equal(outcomes[0][2], outcomes[1][2])
        self.assertEqual(outcomes[0][3:], outcomes[1][3:])

    def test_leaf_grad_identity_and_live_accumulation_match_pytorch_2_13(self):
        outcomes = []
        for module in (torch, reference_torch):
            leaf = module.tensor([2.0, 3.0], requires_grad=True)
            loss = leaf.sum()
            loss.backward()
            retained = leaf.grad
            first_identity = retained is leaf.grad
            first_values = np.asarray(retained).copy()

            loss.backward()
            outcomes.append(
                (
                    first_identity,
                    retained is leaf.grad,
                    first_values,
                    np.asarray(retained).copy(),
                )
            )

        self.assertEqual(outcomes[0][:2], outcomes[1][:2])
        np.testing.assert_array_equal(outcomes[0][2], outcomes[1][2])
        np.testing.assert_array_equal(outcomes[0][3], outcomes[1][3])

    def test_real_scalar_addition_gradients_match_pytorch_2_13(self):
        outcomes = []
        for module in (torch, reference_torch):
            leaf = module.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
            )
            view = leaf.transpose(0, 1)
            weights = module.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            composed = 3.0 + (view + 2.0)
            loss = (composed * weights).sum()
            loss.backward()

            repeated_leaf = module.tensor([2.0, 3.0], requires_grad=True)
            repeated_loss = (repeated_leaf + 1.0).sum()
            repeated_loss.backward()
            repeated_loss.backward()

            empty_leaf = module.tensor(
                np.empty((0,), dtype=np.float32), requires_grad=True
            )
            empty = empty_leaf.reshape(2, 0, 3)
            empty_result = 7.0 + empty
            empty_result.sum().backward()

            detached = leaf.detach() + 1.0
            with module.no_grad():
                suppressed = 1.0 + leaf

            outcomes.append(
                (
                    np.asarray(leaf.grad).copy(),
                    np.asarray(repeated_leaf.grad).copy(),
                    composed.requires_grad,
                    tuple(empty_result.shape),
                    empty_result.requires_grad,
                    tuple(empty_leaf.grad.shape),
                    empty_leaf.grad.numel(),
                    detached.requires_grad,
                    suppressed.requires_grad,
                )
            )

        np.testing.assert_array_equal(outcomes[0][0], outcomes[1][0])
        np.testing.assert_array_equal(outcomes[0][1], outcomes[1][1])
        self.assertEqual(outcomes[0][2:], outcomes[1][2:])

    def test_real_scalar_subtraction_gradients_match_pytorch_2_13(self):
        outcomes = []
        for module in (torch, reference_torch):
            weights = module.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

            forward_leaf = module.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
            )
            forward = forward_leaf.transpose(0, 1) - 2.0
            forward_loss = (forward * weights).sum()
            forward_loss.backward()

            reflected_leaf = module.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
            )
            reflected = 10.0 - reflected_leaf.transpose(0, 1)
            reflected_loss = (reflected * weights).sum()
            reflected_loss.backward()

            repeated_results = []
            for operation in (
                lambda value: value - 1.0,
                lambda value: 1.0 - value,
            ):
                repeated_leaf = module.tensor([2.0, 3.0], requires_grad=True)
                repeated_loss = operation(repeated_leaf).sum()
                repeated_loss.backward()
                repeated_loss.backward()
                repeated_results.append(np.asarray(repeated_leaf.grad).copy())

            empty_results = []
            for operation in (
                lambda value: value - 7.0,
                lambda value: 7.0 - value,
            ):
                empty_leaf = module.tensor(
                    np.empty((0,), dtype=np.float32), requires_grad=True
                )
                empty_output = operation(empty_leaf.reshape(2, 0, 3))
                empty_output.sum().backward()
                empty_results.append(
                    (
                        tuple(empty_output.shape),
                        empty_output.stride(),
                        empty_output.requires_grad,
                        tuple(empty_leaf.grad.shape),
                        empty_leaf.grad.numel(),
                    )
                )

            detached = (
                (forward_leaf.detach() - 1.0).requires_grad,
                (1.0 - forward_leaf.detach()).requires_grad,
            )
            with module.no_grad():
                suppressed = (
                    (forward_leaf - 1.0).requires_grad,
                    (1.0 - forward_leaf).requires_grad,
                )

            outcomes.append(
                (
                    forward.detach().tolist(),
                    forward.stride(),
                    np.asarray(forward_leaf.grad).copy(),
                    reflected.detach().tolist(),
                    reflected.stride(),
                    np.asarray(reflected_leaf.grad).copy(),
                    repeated_results,
                    empty_results,
                    detached,
                    suppressed,
                )
            )

        self.assertEqual(outcomes[0][0:2], outcomes[1][0:2])
        np.testing.assert_array_equal(outcomes[0][2], outcomes[1][2])
        self.assertEqual(outcomes[0][3:5], outcomes[1][3:5])
        np.testing.assert_array_equal(outcomes[0][5], outcomes[1][5])
        for native, expected in zip(outcomes[0][6], outcomes[1][6]):
            np.testing.assert_array_equal(native, expected)
        self.assertEqual(outcomes[0][7:], outcomes[1][7:])

    def test_seeded_square_sum_and_broadcast_gradients_match_pytorch_2_13(self):
        rng = np.random.default_rng(0xA670_213)
        shape_pairs = [
            ((17,), (17,)),
            ((2, 1, 3), (1, 5, 1)),
            ((3, 1), (1, 7)),
            ((), (4, 3)),
            ((0,), (1,)),
        ]
        for left_shape, right_shape in shape_pairs:
            left_values = rng.normal(size=left_shape or ()).astype(np.float32)
            right_values = rng.normal(size=right_shape or ()).astype(np.float32)
            native_left_data = left_values.item() if left_shape == () else left_values.tolist()
            native_right_data = right_values.item() if right_shape == () else right_values.tolist()
            native_left = torch.tensor(native_left_data, requires_grad=True)
            native_right = torch.tensor(native_right_data, requires_grad=True)
            expected_left = reference_torch.tensor(left_values, requires_grad=True)
            expected_right = reference_torch.tensor(right_values, requires_grad=True)

            native_output = (native_left * native_right).sum()
            expected_output = (expected_left * expected_right).sum()
            native_output.backward()
            expected_output.backward()

            with self.subTest(left=left_shape, right=right_shape):
                np.testing.assert_allclose(
                    native_output.item(), expected_output.item(), rtol=1.0e-5, atol=1.0e-6
                )
                self.assert_gradient_pair(native_left, expected_left)
                self.assert_gradient_pair(native_right, expected_right)

        square_values = rng.normal(size=(11, 7)).astype(np.float32)
        native_square = torch.tensor(square_values.tolist(), requires_grad=True)
        expected_square = reference_torch.tensor(square_values, requires_grad=True)
        (native_square * native_square).sum().backward()
        (expected_square * expected_square).sum().backward()
        self.assert_gradient_pair(native_square, expected_square)

    def test_boundaries_scalar_empty_and_errors_match_pytorch_2_13(self):
        accumulated_gradients = []
        for module in (torch, reference_torch):
            scalar = module.tensor(3.0, requires_grad=True)
            (scalar * 5.0).backward()
            self.assertEqual(scalar.grad.item(), 5.0)

            empty = module.tensor(np.empty((0,), dtype=np.float32), requires_grad=True)
            empty.sum().backward()
            self.assertEqual(tuple(empty.grad.shape), (0,))
            self.assertEqual(empty.grad.numel(), 0)

            boundary = module.tensor([2.0], requires_grad=True)
            self.assertFalse((boundary.detach() * boundary.detach()).requires_grad)
            with module.no_grad():
                self.assertFalse((boundary * boundary).requires_grad)
            self.assertTrue((boundary * boundary).requires_grad)

            intermediate = boundary * boundary
            output = intermediate.sum()
            del intermediate
            gc.collect()
            output.backward()
            with self.assertRaises(RuntimeError):
                output.backward()
            (boundary * boundary).sum().backward()
            accumulated_gradients.append(np.asarray(boundary.grad).copy())
            with self.assertRaises(RuntimeError):
                module.tensor([1.0, 2.0], requires_grad=True).backward()
            with self.assertRaisesRegex(RuntimeError, "does not require grad"):
                module.tensor([1.0, 2.0]).backward()
        np.testing.assert_array_equal(accumulated_gradients[0], accumulated_gradients[1])

    def test_transform_and_signed_zero_gradients_match_pytorch_2_13(self):
        gradients = []
        reshape_gradients = []
        signed_zero_bits = []
        broadcast_zero_bits = []
        for module in (torch, reference_torch):
            leaf = module.tensor(
                [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], requires_grad=True
            )
            transformed = leaf.transpose(1, 2).contiguous().squeeze(0)[1].reshape(2, 1).clone()
            (transformed * transformed).sum().backward()
            gradients.append(np.asarray(leaf.grad).copy())

            reshape_leaf = module.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
            )
            reshaped_copy = reshape_leaf.transpose(0, 1).reshape(6)
            weights = module.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
            (reshaped_copy * weights).sum().backward()
            reshape_gradients.append(np.asarray(reshape_leaf.grad).copy())

            signed = module.tensor([2.0], requires_grad=True)
            (signed * module.tensor([-0.0])).sum().backward()
            signed_zero_bits.append(np.asarray(signed.grad).view(np.uint32)[0])

            broadcast_signed = module.tensor(2.0, requires_grad=True)
            (broadcast_signed * module.tensor([-0.0, -0.0])).sum().backward()
            broadcast_zero_bits.append(
                np.asarray(broadcast_signed.grad).view(np.uint32).item()
            )

        np.testing.assert_array_equal(gradients[0], gradients[1])
        np.testing.assert_array_equal(reshape_gradients[0], reshape_gradients[1])
        self.assertEqual(signed_zero_bits, [0x8000_0000, 0x8000_0000])
        self.assertEqual(broadcast_zero_bits, [0, 0])

    def test_no_grad_generator_protocol_matches_pytorch_2_13(self):
        outcomes = []
        for module in (torch, reference_torch):
            value = module.tensor([2.0], requires_grad=True)
            events = []

            @module.no_grad()
            def generate():
                events.append(("next", (value * value).requires_grad))
                request = yield value * value
                events.append(("send", request, (value * value).requires_grad))
                try:
                    yield value * value
                except ValueError as error:
                    events.append(("throw", str(error), (value * value).requires_grad))
                    yield value * value
                finally:
                    events.append(("close", (value * value).requires_grad))

            generator = generate()
            requires_grad = [next(generator).requires_grad]
            requires_grad.append(generator.send("request").requires_grad)
            requires_grad.append(generator.throw(ValueError("injected")).requires_grad)
            generator.close()
            abandoned_events = []

            @module.no_grad()
            def abandoned():
                try:
                    yield value * value
                finally:
                    abandoned_events.append((value * value).requires_grad)

            abandoned_generator = abandoned()
            requires_grad.append(next(abandoned_generator).requires_grad)
            del abandoned_generator
            gc.collect()
            outcomes.append(
                (
                    requires_grad,
                    events,
                    abandoned_events,
                    (value * value).requires_grad,
                )
            )

        self.assertEqual(outcomes[0], outcomes[1])

    def test_no_grad_views_and_public_contract_match_pytorch_2_13(self):
        outcomes = []
        for module in (torch, reference_torch):
            source = module.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
            with module.no_grad() as entered:
                views = [
                    source.transpose(0, 1),
                    source.T,
                    source.reshape(4),
                    source.squeeze(),
                    source[0],
                ]
                materialized = [source.clone(), source.T.contiguous()]

            with self.assertRaisesRegex(RuntimeError, "implicitly created only for scalar"):
                views[2].backward()

            (views[0] * views[0]).sum().backward()

            @module.no_grad
            def decorated(value: object, scale: float = 1.0) -> object:
                """decorated docs"""
                return value * scale

            @module.no_grad()
            def generator():
                yield source * source

            outcomes.append(
                (
                    entered,
                    [view.requires_grad for view in views],
                    [tensor.requires_grad for tensor in materialized],
                    source.grad is None,
                    decorated.__name__,
                    decorated.__doc__,
                    decorated.__annotations__,
                    str(inspect.signature(decorated)),
                    decorated.__wrapped__.__name__,
                    inspect.isgeneratorfunction(generator),
                    next(generator()).requires_grad,
                )
            )

        self.assertEqual(outcomes[0], outcomes[1])

    def test_reusable_metadata_only_graphs_match_pytorch_2_13(self):
        outcomes = []
        for module in (torch, reference_torch):
            summed_leaf = module.tensor([1.0, 2.0], requires_grad=True)
            summed = summed_leaf.sum()
            summed.backward()
            summed.backward()

            transformed_leaf = module.tensor(
                [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
            )
            transformed = transformed_leaf.T.clone().sum()
            transformed.backward()
            transformed.backward()

            multiplied_leaf = module.tensor([1.0, 2.0], requires_grad=True)
            multiplied = (multiplied_leaf * multiplied_leaf).sum()
            multiplied.backward()
            with self.assertRaisesRegex(
                RuntimeError, "backward through the graph a second time"
            ):
                multiplied.backward()

            outcomes.append(
                (
                    np.asarray(summed_leaf.grad).copy(),
                    np.asarray(transformed_leaf.grad).copy(),
                    np.asarray(multiplied_leaf.grad).copy(),
                )
            )

        for native, expected in zip(outcomes[0], outcomes[1]):
            np.testing.assert_array_equal(native, expected)

    def test_square_sum_backward_performance_smoke_is_equivalent_work(self):
        values = np.linspace(-2.0, 2.0, 16_384, dtype=np.float32)

        def measure(module):
            inputs = [module.tensor(values, requires_grad=True) for _ in range(10)]
            samples = []
            checksum = 0.0
            for index, value in enumerate(inputs):
                started = time.perf_counter()
                (value * value).sum().backward()
                elapsed = time.perf_counter() - started
                checksum += float((value.grad * value.grad).sum().item())
                if index >= 3:
                    samples.append(elapsed)
            return statistics.median(samples), checksum

        gc.collect()
        native_seconds, native_checksum = measure(torch)
        gc.collect()
        reference_seconds, reference_checksum = measure(reference_torch)

        self.assertGreater(native_seconds, 0.0)
        self.assertGreater(reference_seconds, 0.0)
        np.testing.assert_allclose(native_checksum, reference_checksum, rtol=1.0e-5)
        self.assertLess(native_seconds / reference_seconds, 250.0)


if __name__ == "__main__":
    unittest.main()
