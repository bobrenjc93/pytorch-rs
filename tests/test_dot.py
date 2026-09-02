import copy
import importlib
import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


METHOD_DOC = "\ndot(other) -> Tensor\n\nSee :func:`torch.dot`\n"
FUNCTION_DOC = """
dot(input, tensor, *, out=None) -> Tensor

Computes the dot product of two 1D tensors.

.. note::

    Unlike NumPy's dot, torch.dot intentionally only supports computing the dot product
    of two 1D tensors with the same number of elements.

Args:
    input (Tensor): first tensor in the dot product, must be 1D.
    tensor (Tensor): second tensor in the dot product, must be 1D.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> torch.dot(torch.tensor([2, 3]), torch.tensor([2, 1]))
    tensor(7)

    >>> t1, t2 = torch.tensor([0, 1]), torch.tensor([2, 3])
    >>> torch.dot(t1, t2)
    tensor(3)
"""


def dot_cases(module):
    dense = module.tensor(
        np.arange(24, dtype=np.float32).reshape(4, 2, 3).tolist(),
        dtype=module.float32,
    )
    offset_left = dense[2][1]
    offset_right = module.tensor(
        [[100.0, 200.0, 300.0], [4.0, -5.0, 6.0]], dtype=module.float32
    )[1]

    strided_base = module.tensor(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
        dtype=module.float32,
    )
    noncontiguous_left = strided_base.transpose(0, 1)[0]
    noncontiguous_right = module.tensor(
        [[-1.0, 9.0], [2.0, 8.0], [-3.0, 7.0], [4.0, 6.0]],
        dtype=module.float32,
    ).transpose(0, 1)[0]

    empty_base = module.zeros((0, 5), dtype=module.float32)
    empty_left = empty_base.transpose(0, 1)[2]
    empty_right = module.ones((5, 0), dtype=module.float32)[3]

    return (
        (
            "contiguous",
            module.tensor([1.0, -2.0, 3.5], dtype=module.float32),
            module.tensor([4.0, 5.0, -6.0], dtype=module.float32),
        ),
        ("offset", offset_left, offset_right),
        ("noncontiguous", noncontiguous_left, noncontiguous_right),
        ("empty", empty_left, empty_right),
        (
            "signed zero",
            module.tensor([-0.0, 0.0, -0.0, 0.0], dtype=module.float32),
            module.tensor([1.0, 1.0, 1.0, 1.0], dtype=module.float32),
        ),
        (
            "nan",
            module.tensor([1.0, float("nan"), 2.0], dtype=module.float32),
            module.tensor([3.0, 4.0, 5.0], dtype=module.float32),
        ),
        (
            "inf",
            module.tensor([float("inf"), 1.0], dtype=module.float32),
            module.tensor([2.0, 3.0], dtype=module.float32),
        ),
        (
            "inf times zero",
            module.tensor([float("inf"), 1.0], dtype=module.float32),
            module.tensor([0.0, 3.0], dtype=module.float32),
        ),
    )


class DotTests(unittest.TestCase):
    def assert_dot_matches_composition(self, actual, expected, left, right, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, ())
            self.assertEqual(actual.stride(), ())
            self.assertEqual(actual.storage_offset(), 0)
            self.assertEqual(actual.numel(), 1)
            self.assertTrue(actual.is_contiguous())
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertFalse(actual.is_set_to(left))
            self.assertFalse(actual.is_set_to(right))
            if left.numel():
                self.assertNotEqual(actual.data_ptr(), left.data_ptr())
            if right.numel():
                self.assertNotEqual(actual.data_ptr(), right.data_ptr())

        with self.subTest(case=case, value=True):
            self.assertEqual(
                np.asarray(actual).view(np.uint32).item(),
                np.asarray(expected).view(np.uint32).item(),
            )

    def test_tensor_and_top_level_forms_match_composed_multiply_sum(self):
        for case, left, right in dot_cases(torch):
            expected = (left * right).sum()
            calls = (
                ("method positional", lambda: left.dot(right)),
                ("method keyword", lambda: left.dot(tensor=right)),
                ("function positional", lambda: torch.dot(left, right)),
                ("function keywords", lambda: torch.dot(input=left, tensor=right)),
                ("function x alias", lambda: torch.dot(x=left, tensor=right)),
                ("function a alias", lambda: torch.dot(a=left, tensor=right)),
                ("function x1 alias", lambda: torch.dot(x1=left, tensor=right)),
                ("function out none", lambda: torch.dot(left, right, out=None)),
            )
            for form, call in calls:
                self.assert_dot_matches_composition(
                    call(), expected, left, right, case=(case, form)
                )

    def test_shape_rank_and_binding_errors_match_the_narrow_surface(self):
        vector = torch.ones((2,))
        other = torch.ones((2,))
        destination = torch.tensor(-9.0)

        cases = (
            (
                lambda: torch.dot(),
                TypeError,
                'dot() missing 2 required positional argument: "input", "tensor"',
            ),
            (
                lambda: torch.dot(vector),
                TypeError,
                'dot() missing 1 required positional arguments: "tensor"',
            ),
            (
                lambda: torch.dot(vector, other, other),
                TypeError,
                "dot() takes 2 positional arguments but 3 were given",
            ),
            (
                lambda: torch.dot([], other),
                TypeError,
                "dot(): argument 'input' (position 1) must be Tensor, not list",
            ),
            (
                lambda: torch.dot(vector, []),
                TypeError,
                "dot(): argument 'tensor' (position 2) must be Tensor, not list",
            ),
            (
                lambda: torch.dot(vector, other, foo=True),
                TypeError,
                "dot() got an unexpected keyword argument 'foo'",
            ),
            (
                lambda: torch.dot(vector, other, input=vector),
                TypeError,
                "dot() got multiple values for argument 'input'",
            ),
            (
                lambda: vector.dot(),
                TypeError,
                'dot() missing 1 required positional arguments: "tensor"',
            ),
            (
                lambda: vector.dot(other, other),
                TypeError,
                "dot() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: vector.dot(input=other),
                TypeError,
                'dot() missing 1 required positional arguments: "tensor"',
            ),
            (
                lambda: vector.dot(other, tensor=other),
                TypeError,
                "dot() got multiple values for argument 'tensor'",
            ),
            (
                lambda: vector.dot(other, out=destination),
                TypeError,
                "dot() got an unexpected keyword argument 'out'",
            ),
            (
                lambda: vector.dot([]),
                TypeError,
                "dot(): argument 'tensor' (position 1) must be Tensor, not list",
            ),
            (
                lambda: torch.dot(torch.tensor(1.0), vector),
                RuntimeError,
                "1D tensors expected, but got 0D and 1D tensors",
            ),
            (
                lambda: torch.dot(torch.ones((2, 1)), vector),
                RuntimeError,
                "1D tensors expected, but got 2D and 1D tensors",
            ),
            (
                lambda: torch.dot(vector, torch.ones((3,))),
                RuntimeError,
                "inconsistent tensor size, expected tensor [2] and src [3] "
                "to have the same number of elements, but got 2 and 3 "
                "elements respectively",
            ),
            (
                lambda: torch.dot(vector, other, out=destination),
                RuntimeError,
                "dot(): the 'out' argument is not supported",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error_type, f"^{re.escape(message)}$"):
                    call()
        self.assertEqual(destination.item(), -9.0)

    def test_no_grad_and_first_order_backward_use_composed_operations(self):
        left_leaf = torch.tensor(
            [[1.0, 99.0], [-2.0, 88.0], [3.0, 77.0], [4.0, 66.0]],
            requires_grad=True,
        )
        right_leaf = torch.tensor(
            [[5.0, 55.0], [6.0, 44.0], [-7.0, 33.0], [8.0, 22.0]],
            requires_grad=True,
        )
        left = left_leaf.transpose(0, 1)[0]
        right = right_leaf.transpose(0, 1)[0]

        loss = torch.dot(left, right)
        self.assertTrue(loss.requires_grad)
        self.assertFalse(loss.is_leaf)
        loss.backward()

        expected_left_grad = np.zeros((4, 2), dtype=np.float32)
        expected_left_grad[:, 0] = np.asarray(right.detach())
        expected_right_grad = np.zeros((4, 2), dtype=np.float32)
        expected_right_grad[:, 0] = np.asarray(left.detach())
        np.testing.assert_array_equal(np.asarray(left_leaf.grad), expected_left_grad)
        np.testing.assert_array_equal(np.asarray(right_leaf.grad), expected_right_grad)

        empty_left = torch.zeros((0,), requires_grad=True)
        empty_right = torch.zeros((0,), requires_grad=True)
        empty_loss = empty_left.dot(empty_right)
        self.assertTrue(empty_loss.requires_grad)
        empty_loss.backward()
        self.assertEqual(empty_left.grad.tolist(), [])
        self.assertEqual(empty_right.grad.tolist(), [])

        with torch.no_grad():
            untracked = torch.dot(left, right)
        self.assertFalse(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)

    def test_callable_metadata_import_copy_pickle_and_unsupported_aliases(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "dot")
        bound = tensor.dot

        self.assertIs(getattr(torch.Tensor, "dot"), descriptor)
        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(repr(descriptor), "<method 'dot' of 'torch._C.TensorBase' objects>")
        self.assertEqual(descriptor.__name__, "dot")
        self.assertEqual(descriptor.__qualname__, "TensorBase.dot")
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertEqual(bound.__name__, "dot")
        self.assertEqual(bound.__qualname__, "Tensor.dot")
        self.assertIsNone(bound.__module__)
        self.assertNotIn("dot", torch.Tensor.__dict__)
        self.assertEqual(descriptor.__doc__, METHOD_DOC)
        self.assertEqual(bound.__doc__, METHOD_DOC)

        for callable_object in (descriptor, bound):
            self.assertIsNone(callable_object.__text_signature__)
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)
            self.assertIs(copy.copy(callable_object), callable_object)
            self.assertIs(copy.deepcopy(callable_object), callable_object)

        function = torch.dot
        self.assertTrue(callable(function))
        self.assertEqual(function.__name__, "dot")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.dot")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertEqual(torch.__all__.count("dot"), 1)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)

        self.assert_dot_matches_composition(
            descriptor(tensor, tensor),
            tensor.dot(tensor),
            tensor,
            tensor,
            case="unbound positional",
        )
        with self.assertRaisesRegex(
            TypeError,
            r"^unbound method TensorBase\.dot\(\) needs an argument$",
        ):
            descriptor()
        with self.assertRaisesRegex(
            TypeError,
            r"^descriptor 'dot' for 'torch\._C\.TensorBase' objects "
            r"doesn't apply to a 'int' object$",
        ):
            descriptor(1, tensor)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol, callable="function"):
                self.assertIs(pickle.loads(pickle.dumps(function, protocol)), function)
            with self.subTest(protocol=protocol, callable="descriptor"):
                self.assertIs(pickle.loads(pickle.dumps(descriptor, protocol)), descriptor)

        reloaded = importlib.reload(torch)
        self.assertIs(reloaded, torch)
        self.assertIs(torch.dot, function)
        self.assertIs(inspect.getattr_static(torch.Tensor, "dot"), descriptor)
        self.assertFalse(hasattr(torch, "vdot"))
        self.assertFalse(hasattr(torch, "inner"))
        self.assertFalse(hasattr(torch, "outer"))

    def test_torch_function_dispatch_boundaries(self):
        left = torch.tensor([1.0, 2.0])
        right = torch.tensor([3.0, 4.0])
        descriptor = inspect.getattr_static(torch.Tensor, "dot")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode_calls = (
            ("method positional", lambda: left.dot(right), descriptor, 2, None),
            ("method keyword", lambda: left.dot(tensor=right), descriptor, 1, ("tensor",)),
            ("function positional", lambda: torch.dot(left, right), torch.dot, 2, None),
            (
                "function keywords",
                lambda: torch.dot(input=left, tensor=right),
                torch.dot,
                0,
                ("input", "tensor"),
            ),
        )
        for case, call, expected_function, expected_arg_count, expected_keywords in mode_calls:
            mode = RecordingMode()
            with mode:
                self.assertIs(call(), marker)
            self.assertEqual(len(mode.calls), 1)
            function, dispatch_types, args, kwargs = mode.calls[0]
            with self.subTest(case=case):
                self.assertIs(function, expected_function)
                self.assertEqual(dispatch_types, ())
                self.assertEqual(len(args), expected_arg_count)
                if expected_keywords is None:
                    self.assertIsNone(kwargs)
                else:
                    self.assertEqual(tuple(kwargs), expected_keywords)

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        for case, call, expected_function, expected_types in (
            (
                "method other",
                lambda value: left.dot(value),
                descriptor,
                (Override,),
            ),
            (
                "function input",
                lambda value: torch.dot(value, right),
                torch.dot,
                (Override,),
            ),
            (
                "function tensor",
                lambda value: torch.dot(left, value),
                torch.dot,
                (Override,),
            ),
            (
                "function out",
                lambda value: torch.dot(left, right, out=value),
                torch.dot,
                (Override,),
            ),
        ):
            value = Override()
            Override.calls.clear()
            self.assertIs(call(value), marker)
            function, dispatch_types, args, kwargs = Override.calls[0]
            with self.subTest(case=case):
                self.assertIs(function, expected_function)
                self.assertEqual(dispatch_types, expected_types)
                self.assertTrue(args or kwargs)

        events = []

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append("override")
                return marker

        declining_mode = RecordingMode(NotImplemented)
        with declining_mode:
            self.assertIs(torch.dot(input=left, tensor=DecliningOverride()), marker)
        self.assertEqual(len(declining_mode.calls), 1)
        self.assertEqual(events, ["override"])

        class OnlyDeclines:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaises(TypeError) as raised:
            torch.dot(OnlyDeclines(), OnlyDeclines())
        self.assertIn("Multiple dispatch failed for 'torch.dot'", str(raised.exception))

        invalid_mode = RecordingMode()
        with invalid_mode:
            with self.assertRaises(TypeError):
                torch.dot([], right)
        self.assertEqual(invalid_mode.calls, [])

    def test_related_linear_algebra_and_dtype_device_expansion_stay_out_of_scope(self):
        vector = torch.ones((2,))
        with self.assertRaisesRegex(RuntimeError, "requires two rank-2 tensors"):
            torch.matmul(vector, vector)
        with self.assertRaisesRegex(RuntimeError, "requires two rank-2 tensors"):
            vector @ vector

        self.assertFalse(hasattr(torch, "vdot"))
        self.assertFalse(hasattr(torch, "inner"))
        self.assertFalse(hasattr(torch, "outer"))
        self.assertFalse(hasattr(torch, "complex64"))
        self.assertFalse(hasattr(torch, "float64"))


if __name__ == "__main__":
    unittest.main()
