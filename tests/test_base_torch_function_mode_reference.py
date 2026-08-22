import copy
import inspect
import pickle
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class BaseTorchFunctionModeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "BaseTorchFunctionMode differentials require pinned PyTorch 2.13.0"
            )

    def metadata_observation(self, module):
        mode_type = module.overrides.BaseTorchFunctionMode
        namespace = {}
        exec(f"from {module.overrides.__name__} import *", namespace)
        constructor_errors = []
        for constructor in (
            lambda: mode_type(1),
            lambda: mode_type(value=1),
        ):
            try:
                constructor()
            except Exception as error:
                constructor_errors.append(
                    (type(error).__name__, str(error), error.args)
                )
            else:
                constructor_errors.append(None)

        instance = mode_type()
        return (
            mode_type.__name__,
            mode_type.__qualname__,
            mode_type.__module__.rsplit(".", 1)[-1],
            mode_type.__bases__[0].__name__,
            issubclass(mode_type, module.overrides.TorchFunctionMode),
            mode_type.__doc__,
            str(inspect.signature(mode_type)),
            str(inspect.signature(mode_type.__torch_function__)),
            mode_type.__torch_function__.__defaults__,
            mode_type.__torch_function__.__annotations__,
            "BaseTorchFunctionMode" in module.overrides.__all__,
            "BaseTorchFunctionMode" in namespace,
            type(instance).__name__,
            instance.__dict__,
            constructor_errors,
        )

    def test_construction_metadata_and_exports_match_pytorch_2_13(self):
        self.assertEqual(
            self.metadata_observation(torch),
            self.metadata_observation(reference_torch),
        )

    def copy_observation(self, module):
        mode = module.overrides.BaseTorchFunctionMode()
        mode.payload = [1, 2, 3]
        shallow = copy.copy(mode)
        deep = copy.deepcopy(mode)
        pickle_results = []
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            restored = pickle.loads(pickle.dumps(mode, protocol=protocol))
            pickle_results.append(
                (
                    type(restored).__name__,
                    restored is mode,
                    restored.__dict__,
                    restored.payload is mode.payload,
                )
            )
        return (
            type(shallow).__name__,
            shallow is mode,
            shallow.__dict__,
            shallow.payload is mode.payload,
            type(deep).__name__,
            deep is mode,
            deep.__dict__,
            deep.payload is mode.payload,
            pickle_results,
        )

    def test_copying_and_pickling_match_pytorch_2_13(self):
        self.assertEqual(
            self.copy_observation(torch),
            self.copy_observation(reference_torch),
        )

    def direct_handler_observation(self, module):
        mode = module.overrides.BaseTorchFunctionMode()
        marker = object()
        calls = []

        def target(*args, **kwargs):
            calls.append((args, kwargs))
            return marker

        first = mode.__torch_function__(target, (str,), (1, 2), None)
        second = mode.__torch_function__(target, (), (), {"answer": 42})
        error = RuntimeError("target failed")

        def raising_target():
            raise error

        try:
            mode.__torch_function__(raising_target, (), (), None)
        except Exception as raised:
            error_result = (
                type(raised).__name__,
                str(raised),
                raised.args,
                raised is error,
            )
        else:
            error_result = None
        return first is marker, second is marker, calls, error_result

    def test_direct_forwarding_matches_pytorch_2_13(self):
        self.assertEqual(
            self.direct_handler_observation(torch),
            self.direct_handler_observation(reference_torch),
        )

    def dispatch_observation(self, module):
        tensor = module.tensor([[1.0, 2.0], [3.0, 4.0]])
        expected_adjoint = tensor.mH
        events = []

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                events.append(
                    (
                        func.__name__,
                        tuple(dispatch_type.__name__ for dispatch_type in types),
                        tuple(argument is tensor for argument in args),
                        kwargs,
                        len(module.overrides._get_current_function_mode_stack()),
                    )
                )
                return func(*args, **(kwargs or {}))

        lower = RecordingMode()
        upper = module.overrides.BaseTorchFunctionMode()
        with lower:
            with upper:
                positive = module.positive(tensor)
                adjoint = tensor.adjoint()
                real = tensor.real
                restored_inside = (
                    module.overrides._get_current_function_mode_stack()
                    == [lower, upper]
                )

        return (
            positive is tensor,
            adjoint.is_set_to(expected_adjoint),
            tuple(adjoint.shape),
            tuple(adjoint.stride()),
            real is tensor,
            events,
            restored_inside,
            module.overrides._get_current_function_mode_stack() == [],
        )

    def test_nested_function_method_and_property_forwarding_matches(self):
        self.assertEqual(
            self.dispatch_observation(torch),
            self.dispatch_observation(reference_torch),
        )

    def exception_observation(self, module):
        tensor = module.tensor([1.0])
        base_mode = module.overrides.BaseTorchFunctionMode()
        with base_mode:
            try:
                tensor.H
            except Exception as error:
                native_error = (type(error).__name__, str(error), error.args)
            else:
                native_error = None
            native_restored = (
                module.overrides._get_current_function_mode_stack() == [base_mode]
            )

        expected_error = ValueError("lower mode failed")

        class RaisingMode(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                raise expected_error

        lower = RaisingMode()
        with lower:
            with base_mode:
                try:
                    module.positive(tensor)
                except Exception as error:
                    lower_error = (
                        type(error).__name__,
                        str(error),
                        error.args,
                        error is expected_error,
                    )
                else:
                    lower_error = None
                nested_restored = (
                    module.overrides._get_current_function_mode_stack()
                    == [lower, base_mode]
                )
            lower_restored = (
                module.overrides._get_current_function_mode_stack() == [lower]
            )
        return (
            native_error,
            native_restored,
            lower_error,
            nested_restored,
            lower_restored,
            module.overrides._get_current_function_mode_stack() == [],
        )

    def test_exception_propagation_and_stack_restoration_match(self):
        self.assertEqual(
            self.exception_observation(torch),
            self.exception_observation(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
