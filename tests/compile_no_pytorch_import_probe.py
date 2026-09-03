import sys


class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None


sys.meta_path.insert(0, RejectPytorchImport())

import torch_rs as torch


def relu_model(x):
    return x.relu()


captured_argument_types = []


def capture_trace(frame, event, arg):
    if frame.f_code is relu_model.__code__ and event == "line":
        value = frame.f_locals.get("x")
        captured_argument_types.append(
            (
                type(value).__module__,
                type(value).__name__,
                type(value) is torch.Tensor,
            )
        )
    return capture_trace


sys.settrace(capture_trace)
try:
    compiled = torch.compile(relu_model, backend="eager")
finally:
    sys.settrace(None)

assert captured_argument_types == [
    ("torch_rs._compile", "_ReluTraceProxy", False)
], captured_argument_types

model_executions = []


def execution_profile(frame, event, arg):
    if event == "call" and frame.f_code is relu_model.__code__:
        model_executions.append("model")


sys.setprofile(execution_profile)
try:
    output = compiled(torch.tensor([-1.0, 0.0, 2.5]))
finally:
    sys.setprofile(None)

assert model_executions == [], model_executions
assert output.tolist() == [0.0, 0.0, 2.5], output.tolist()
assert not any(
    name == "torch" or name.startswith("torch.") for name in sys.modules
), "installed PyTorch appeared in sys.modules"
