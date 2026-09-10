"""Public linear mode dispatch, checked against the pinned PyTorch release."""

import copy
import itertools
import pickle
import unittest

import numpy as np
import torch_rs

try:
    import torch as reference
except ImportError:
    reference = None


@unittest.skipIf(reference is None, "install the reference dependency group")
class LinearModeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("linear mode differentials require PyTorch 2.13.0")

    def operands(self, module, shape):
        values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        return (
            module.tensor(values.tolist(), dtype=module.float32).reshape(shape),
            module.tensor([[1., -2., 3.], [-4., 5., 6.]]),
            module.tensor([0.5, -1.5]),
        )

    @staticmethod
    def arguments(operands, form):
        x, w, b = operands
        return {
            "omitted": ((x, w), {}),
            "positional none": ((x, w, None), {}),
            "keyword none": ((x, w), {"bias": None}),
            "positional bias": ((x, w, b), {}),
            "keyword bias": ((x, w), {"bias": b}),
            "keyword weight": ((x,), {"weight": w, "bias": b}),
            "all keywords": ((), {"bias": b, "weight": w, "input": x}),
        }[form]

    def assert_call(self, module, func, types, args, kwargs, expected):
        self.assertIs(func, module.nn.functional.linear)
        self.assertEqual(types, ())
        expected_args, expected_kwargs = expected
        self.assertEqual(len(args), len(expected_args))
        for actual, wanted in zip(args, expected_args):
            self.assertIs(actual, wanted)
        self.assertEqual(tuple(kwargs), tuple(expected_kwargs))
        for key in kwargs:
            self.assertIs(kwargs[key], expected_kwargs[key])

    def run_modes(self, module, shape, form, intercept):
        operands = self.operands(module, shape)
        args, kwargs = self.arguments(operands, form)
        events = []
        sentinel = object()
        test = self

        class Mode(module.overrides.TorchFunctionMode):
            def __init__(self, name):
                self.name = name

            def __torch_function__(self, func, types, args=(), kwargs=None):
                test.assert_call(module, func, types, args, kwargs, expected)
                stack = module.overrides._get_current_function_mode_stack()
                events.append((self.name, tuple(mode.name for mode in stack)))
                if intercept:
                    return sentinel
                return func(*args, **(kwargs or {}))

        expected = args, kwargs
        with Mode("outer"), Mode("inner"):
            result = module.nn.functional.linear(*args, **kwargs)
        self.assertEqual(module.overrides._get_current_function_mode_stack(), [])
        if intercept:
            self.assertIs(result, sentinel)
            self.assertEqual(events, [("inner", ("outer",))])
            return events, None
        self.assertEqual(events, [("inner", ("outer",)), ("outer", ())])
        self.assertEqual(result.storage_offset(), 0)
        self.assertTrue(result.is_contiguous())
        self.assertFalse(result.requires_grad)
        # No mode sees implementation details such as transpose or matmul.
        return events, (result.tolist(), tuple(result.shape), result.stride())

    def test_interception_and_nested_delegation_preserve_call(self):
        for shape, form, intercept in itertools.product(
            ((3,), (2, 3), (2, 2, 3), (0, 3), (2, 0, 3)),
            ("omitted", "positional none", "keyword none", "positional bias",
             "keyword bias", "keyword weight", "all keywords"),
            (False, True),
        ):
            with self.subTest(shape=shape, form=form, intercept=intercept):
                self.assertEqual(
                    self.run_modes(torch_rs, shape, form, intercept),
                    self.run_modes(reference, shape, form, intercept),
                )

    @staticmethod
    def positional_call(linear, operands, bias_form, expand_empty):
        x, w, b = operands
        if bias_form == "omitted":
            if expand_empty:
                return linear(x, w, **{})
            return linear(x, w)
        bias = None if bias_form == "none" else b
        if expand_empty:
            return linear(x, w, bias, **{})
        return linear(x, w, bias)

    def keyword_presence(self, module, bias_form, expand_empty, outcome):
        operands = self.operands(module, (2, 3))
        events = []
        test = self

        class Mode(module.overrides.TorchFunctionMode):
            def __init__(self, name):
                self.name = name

            def __repr__(self):
                return self.name

            def __torch_function__(self, func, types, args=(), kwargs=None):
                test.assertIs(func, module.nn.functional.linear)
                test.assertEqual(types, ())
                test.assertEqual(len(args), 2 if bias_form == "omitted" else 3)
                test.assertIs(args[0], operands[0])
                test.assertIs(args[1], operands[1])
                if len(args) == 3:
                    test.assertIs(args[2], None if bias_form == "none" else operands[2])
                if expand_empty:
                    test.assertEqual(kwargs, {})
                else:
                    test.assertIsNone(kwargs)
                events.append((self.name, kwargs is None))
                if outcome == "intercept":
                    return "intercepted"
                if outcome == "decline":
                    return NotImplemented
                if kwargs is None:
                    return func(*args)
                return func(*args, **kwargs)

        with Mode("outer"), Mode("inner"):
            if outcome == "decline":
                with self.assertRaises(TypeError) as raised:
                    self.positional_call(module.nn.functional.linear, operands, bias_form, expand_empty)
                result = str(raised.exception)
            else:
                result = self.positional_call(module.nn.functional.linear, operands, bias_form, expand_empty)
        self.assertEqual(module.overrides._get_current_function_mode_stack(), [])
        self.assertEqual([name for name, _ in events], ["inner", "outer"] if outcome == "delegate" else ["inner"])
        return events, result.tolist() if outcome == "delegate" else result

    def test_direct_positional_calls_distinguish_omitted_and_empty_keywords(self):
        for bias_form, expand_empty, outcome in itertools.product(
            ("omitted", "none", "tensor"), (False, True), ("intercept", "delegate", "decline")
        ):
            with self.subTest(bias=bias_form, empty_keywords=expand_empty, outcome=outcome):
                self.assertEqual(
                    self.keyword_presence(torch_rs, bias_form, expand_empty, outcome),
                    self.keyword_presence(reference, bias_form, expand_empty, outcome),
                )

    def test_three_argument_hooks_intercept_delegate_and_recover(self):
        for bias_form, delegate in itertools.product(("omitted", "none", "tensor"), (False, True)):
            results = []
            for module in (torch_rs, reference):
                operands = self.operands(module, (2, 3))
                test = self

                class Mode(module.overrides.TorchFunctionMode):
                    def __torch_function__(self, func, types, args=()):
                        test.assertIs(func, module.nn.functional.linear)
                        test.assertEqual(types, ())
                        return func(*args) if delegate else "intercepted"

                outer, inner = Mode(), Mode()
                with outer, inner:
                    # Supplying even an empty dictionary must still call the
                    # hook with four arguments and raise for this narrow hook.
                    with self.assertRaises(TypeError):
                        self.positional_call(module.nn.functional.linear, operands, bias_form, True)
                    self.assertEqual(module.overrides._get_current_function_mode_stack(), [outer, inner])
                    result = self.positional_call(module.nn.functional.linear, operands, bias_form, False)
                self.assertEqual(module.overrides._get_current_function_mode_stack(), [])
                results.append(result.tolist() if delegate else result)
            with self.subTest(bias=bias_form, delegate=delegate):
                self.assertEqual(*results)

    def test_native_entry_point_argument_binding_errors_match(self):
        observations = []
        for module in (torch_rs, reference):
            x, w, b = self.operands(module, (2, 3))
            errors = []
            for args, kwargs in (
                ((), {}),
                ((x,), {}),
                ((), {"weight": w}),
                ((x, w, b, None), {}),
                ((x, w), {"input": x}),
                ((x, w), {"weight": w}),
                ((x, w, b), {"bias": b}),
                ((x, w), {"unexpected": None}),
            ):
                with module.overrides.BaseTorchFunctionMode():
                    with self.assertRaises(TypeError) as raised:
                        module.nn.functional.linear(*args, **kwargs)
                errors.append(str(raised.exception))
            observations.append(errors)
        self.assertEqual(*observations)

    def recovery(self, module, failure):
        x, w, b = self.operands(module, (2, 3))
        bad_weight = module.ones((2, 4))
        events = []
        test = self

        class Mode(module.overrides.TorchFunctionMode):
            def __init__(self, name):
                self.name = name
                self.fail = False

            def __repr__(self):
                return self.name

            def __torch_function__(self, func, types, args=(), kwargs=None):
                test.assertIs(func, module.nn.functional.linear)
                test.assertEqual(types, ())
                events.append(self.name)
                if self.fail and failure == "hook":
                    raise ValueError("linear mode failure")
                if self.fail and failure == "NotImplemented":
                    return NotImplemented
                return func(*args, **(kwargs or {}))

        outer, inner = Mode("outer"), Mode("inner")
        with outer:
            with inner:
                inner.fail = True
                with self.assertRaises(Exception) as raised:
                    module.nn.functional.linear(
                        x, bad_weight if failure == "native" else w, bias=b
                    )
                error = type(raised.exception).__name__, str(raised.exception)
                self.assertEqual(events, ["inner", "outer"] if failure == "native" else ["inner"])
                self.assertEqual(module.overrides._get_current_function_mode_stack(), [outer, inner])
                inner.fail = False
                module.nn.functional.linear(input=x, weight=w, bias=b)
                self.assertEqual(events[-2:], ["inner", "outer"])
            module.nn.functional.linear(x, w, b)
            self.assertEqual(events[-1], "outer")
        self.assertEqual(module.overrides._get_current_function_mode_stack(), [])
        # Also let the exception escape both context managers entirely.
        inner.fail = True
        with self.assertRaises(type(raised.exception)):
            with outer, inner:
                module.nn.functional.linear(
                    x, bad_weight if failure == "native" else w, bias=b
                )
        self.assertEqual(module.overrides._get_current_function_mode_stack(), [])
        result = module.nn.functional.linear(x, w, b)
        return error, events, result.tolist()

    def test_exception_notimplemented_and_native_error_recovery(self):
        for failure in ("hook", "NotImplemented", "native"):
            with self.subTest(failure=failure):
                self.assertEqual(self.recovery(torch_rs, failure), self.recovery(reference, failure))

    def test_base_mode_no_grad_and_callable_metadata(self):
        for module in (torch_rs, reference):
            linear = module.nn.functional.linear
            self.assertIs(copy.copy(linear), linear)
            self.assertIs(pickle.loads(pickle.dumps(linear)), linear)
            for tracked in range(3):
                operands = list(self.operands(module, (2, 2, 3)))
                operands[tracked].requires_grad_(True)
                with module.no_grad(), module.overrides.BaseTorchFunctionMode():
                    result = linear(*operands)
                self.assertEqual(result.tolist(), linear(*(x.detach() for x in operands)).tolist())
                self.assertFalse(result.requires_grad)

    def test_mode_can_replace_keyword_bias_before_delegating_strided_operands(self):
        results = []
        for module in (torch_rs, reference):
            # Offset input plus strided weight and bias exercise the existing
            # native layout paths after a mode changes the requested bias.
            x = module.tensor([[[99., 99., 99.]], [[1., 2., 3.]]])[1]
            w = module.tensor([[1., -4.], [-2., 5.], [3., 6.]]).transpose(0, 1)
            replacement = module.tensor([[99., 0.5], [99., -1.5]]).transpose(0, 1)[1]
            call_kwargs = {"bias": None}
            test = self

            class ReplaceBias(module.overrides.TorchFunctionMode):
                def __torch_function__(self, func, types, args=(), kwargs=None):
                    test.assertIs(func, module.nn.functional.linear)
                    test.assertIsNone(kwargs["bias"])
                    kwargs["bias"] = replacement
                    return func(*args, **kwargs)

            with ReplaceBias():
                result = module.nn.functional.linear(x, w, **call_kwargs)
            self.assertEqual(call_kwargs, {"bias": None})
            self.assertEqual(result.tolist(), module.nn.functional.linear(x, w, replacement).tolist())
            results.append(result.tolist())
        self.assertEqual(*results)

    def test_delegation_preserves_native_rejections_and_foreign_boundary(self):
        module = torch_rs
        x, w, b = self.operands(module, (2, 3))
        cases = [
            (module.ones((1, 1, 2, 3)), w, b),
            (module.ones((2, 3, 2)).transpose(1, 2), w, b),
            (x, w, module.ones((2, 2))),
            (x, w, module.ones((4,))),
        ]
        for tracked in range(3):
            operands = list(self.operands(module, (2, 3)))
            operands[tracked].requires_grad_(True)
            cases.append(tuple(operands))
        for operands in cases:
            with self.subTest(shapes=[tuple(x.shape) for x in operands]):
                with self.assertRaises(Exception) as plain:
                    module.nn.functional.linear(*operands)
                with module.overrides.BaseTorchFunctionMode():
                    with self.assertRaises(type(plain.exception)) as delegated:
                        module.nn.functional.linear(*operands)
                self.assertEqual(str(delegated.exception), str(plain.exception))

        class Foreign:
            @classmethod
            def __torch_function__(cls, *args, **kwargs):
                self.fail("foreign tensor handlers remain unsupported")

        for operands in ((Foreign(), w, b), (x, Foreign(), b), (x, w, Foreign())):
            with module.overrides.BaseTorchFunctionMode():
                with self.assertRaisesRegex(TypeError, "only supports.*exact native Tensor"):
                    module.nn.functional.linear(*operands)

    @unittest.skipUnless(reference is not None and reference.cuda.is_available(), "requires NVIDIA GPU")
    def test_cuda_delegation_keeps_native_device_boundary(self):
        # Execute the reference operation on real hardware, then check that a
        # torch_rs mode cannot accidentally enable unsupported native linear.
        x, w, b = (operand.cuda() for operand in self.operands(reference, (2, 3)))
        with reference.overrides.BaseTorchFunctionMode():
            expected = reference.nn.functional.linear(x, w, bias=b)
        reference.cuda.synchronize()
        self.assertEqual(expected.cpu().tolist(), [[4.5, 15.5], [10.5, 36.5]])
        native_x = torch_rs.tensor([[0., 1., 2.], [3., 4., 5.]]).to("cuda:0")
        _, native_w, native_b = self.operands(torch_rs, (2, 3))
        with self.assertRaises(Exception) as plain:
            torch_rs.nn.functional.linear(native_x, native_w, native_b)
        with torch_rs.overrides.BaseTorchFunctionMode():
            with self.assertRaises(type(plain.exception)) as delegated:
                torch_rs.nn.functional.linear(native_x, native_w, bias=native_b)
        self.assertEqual(str(delegated.exception), str(plain.exception))


if __name__ == "__main__":
    unittest.main()
