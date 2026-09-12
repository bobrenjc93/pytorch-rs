"""Public-API programs for default compile parity, independent of torch_rs internals.

This is a finite, versioned sample, not a probability distribution over all Python.
Factories run unchanged with either framework. Inputs never carry compiler markers.
"""

from dataclasses import dataclass


VERSION = "public-default-compile-v2"
CATEGORY_WEIGHTS = {
    "tensor_arithmetic": 12,
    "broadcasting": 8,
    "modules_parameters_buffers": 8,
    "inference": 6,
    "training_autograd": 8,
    "python_control_flow": 8,
    "graph_breaks": 8,
    "dynamic_shapes": 8,
    "mutation_aliasing_views": 8,
    "containers_pytrees": 6,
    "decompositions": 6,
    "custom_functions": 6,
    "recompilation_guards": 4,
    "dtype_device_transitions": 4,
}
VARIANTS = (0, 1)


@dataclass(frozen=True)
class Case:
    name: str
    category: str
    training: bool = False
    observe_inputs: bool = False
    alias_probe: bool = False


CASES = (
    Case("affine_relu", "tensor_arithmetic"),
    Case("trigonometric", "tensor_arithmetic"),
    Case("row_broadcast", "broadcasting"),
    Case("rank3_broadcast", "broadcasting"),
    Case("linear_module", "modules_parameters_buffers"),
    Case("parameter_and_buffer", "modules_parameters_buffers"),
    Case("mlp", "inference"),
    Case("attention", "inference"),
    Case("matmul_backward", "training_autograd", training=True),
    Case("mlp_backward", "training_autograd", training=True),
    Case("shape_branch", "python_control_flow"),
    Case("static_loop", "python_control_flow"),
    Case("scalar_graph_break", "graph_breaks"),
    Case("python_graph_break", "graph_breaks"),
    Case("shape_reshape", "dynamic_shapes"),
    Case("shape_reduction", "dynamic_shapes"),
    Case(
        "mutating_view",
        "mutation_aliasing_views",
        observe_inputs=True,
        alias_probe=True,
    ),
    Case(
        "transpose_view",
        "mutation_aliasing_views",
        observe_inputs=True,
        alias_probe=True,
    ),
    Case("nested_outputs", "containers_pytrees"),
    Case("nested_inputs", "containers_pytrees"),
    Case("layer_norm", "decompositions"),
    Case("gelu", "decompositions"),
    Case("python_helper", "custom_functions"),
    Case("custom_autograd", "custom_functions", training=True),
    Case("scalar_guard", "recompilation_guards"),
    Case("module_attribute_guard", "recompilation_guards"),
    Case("bfloat16_roundtrip", "dtype_device_transitions"),
    Case("device_roundtrip", "dtype_device_transitions"),
)


@dataclass
class Program:
    function: object
    inputs: object
    parameters: tuple = ()


def build(case, fw, device):
    """Return a program and a fresh-input factory; setup is never timed."""
    import numpy as np

    input_epoch = 0
    public_device = "cuda:0" if device == "cuda" else "cpu"

    def tensor(shape, salt, variant, grad=False):
        # Neither framework RNG nor installation order affects shared inputs.
        rng = np.random.default_rng(
            1729 + salt * 101 + variant * 1009 + input_epoch * 1000003
        )
        values = rng.normal(0, 0.2, shape).astype("float32").tolist()
        value = fw.tensor(values, dtype=fw.float32).to(device=public_device)
        return value.requires_grad_(True) if grad else value

    def matrix(variant, grad=False):
        return tensor((128 if variant == 0 else 193, 256), 1, variant, grad)

    def ordinary_inputs(variant):
        return (matrix(variant),)

    def linear(in_features, out_features, salt):
        module = fw.nn.Linear(
            in_features, out_features, device=public_device, dtype=fw.float32
        )
        module.weight = fw.nn.Parameter(tensor((out_features, in_features), salt, 0))
        module.bias = fw.nn.Parameter(tensor((out_features,), salt + 1, 0))
        return module

    name = case.name
    parameters = ()
    inputs = ordinary_inputs
    if name == "affine_relu":

        def program(x):
            return (x * 1.75 - 0.125).relu()
    elif name == "trigonometric":

        def program(x):
            return (x.sin() + x.cos()) * (x + 0.5)
    elif name == "row_broadcast":

        def program(x, bias):
            return (x + bias).relu()

        def inputs(v):
            return matrix(v), tensor((256,), 2, v)
    elif name == "rank3_broadcast":

        def program(x, scale):
            return x * scale + scale

        def inputs(v):
            return tensor((4 + v, 32, 64), 1, v), tensor((1, 32, 1), 2, v)
    elif name in {"linear_module", "mlp", "mlp_backward"}:
        if name == "linear_module":
            program = linear(64, 32, 3)
        else:
            program = fw.nn.Sequential(
                linear(64, 128, 3), fw.nn.ReLU(), linear(128, 16, 5)
            )
        parameters = tuple(program.parameters())

        def inputs(v):
            return (tensor((128 if v == 0 else 193, 64), 1, v, case.training),)
    elif name in {"parameter_and_buffer", "module_attribute_guard"}:

        class Affine(fw.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = fw.nn.Parameter(tensor((256,), 3, 0))
                self.register_buffer("offset", tensor((256,), 4, 0))
                self.scale = 1.0

            def forward(self, x):
                return (x * self.weight + self.offset) * self.scale

        program = Affine()
        parameters = tuple(program.parameters())

        def inputs(v):
            if name == "module_attribute_guard":
                program.scale = 1.0 if v == 0 else 2.5
            return (matrix(v),)
    elif name == "attention":

        def program(q, k, v):
            return fw.nn.functional.scaled_dot_product_attention(q, k, v)

        def inputs(v):
            shape = (2, 4, 64 if v == 0 else 96, 32)
            return tuple(tensor(shape, salt, v) for salt in (1, 2, 3))
    elif name == "matmul_backward":

        def program(x, weight):
            return (x @ weight).sin()

        def inputs(v):
            return tensor((128 + v * 65, 128), 1, v, True), tensor(
                (128, 64), 2, v, True
            )
    elif name == "shape_branch":

        def program(x):
            if x.shape[0] > 150:
                return x.sin() + 0.75
            return x.cos() - 0.25
    elif name == "static_loop":

        def program(x):
            for _ in range(3):
                x = (x * 0.5 + 0.1).relu()
            return x
    elif name == "scalar_graph_break":

        def program(x):
            y = x.sin()
            if y.sum().item() > 0:
                return y * 2.0
            return y - 0.5
    elif name == "python_graph_break":

        def program(x):
            y = x * 1.5
            # Public Python materialization between two tensor regions.
            offset = float(y[0, 0].item())
            return y.cos() + offset
    elif name == "shape_reshape":

        def program(x):
            return x.reshape(x.shape[0], 16, -1).sum(dim=1)
    elif name == "shape_reduction":

        def program(x):
            return x.sum(dim=0) / x.shape[0]
    elif name == "mutating_view":

        def program(x):
            y = x.view(-1)
            y.add_(0.25)
            return y, x
    elif name == "transpose_view":

        def program(x):
            return x.transpose(0, 1), x
    elif name == "nested_outputs":

        def program(x):
            y = x.relu()
            return {"values": (y, [x.sin(), y]), "rows": x.shape[0]}
    elif name == "nested_inputs":

        def program(items):
            return items["x"].sin() + items["extras"][0]

        def inputs(v):
            return ({"x": matrix(v), "extras": [tensor((256,), 2, v)]},)
    elif name == "layer_norm":

        def program(x):
            return fw.nn.functional.layer_norm(x, (256,))
    elif name == "gelu":

        def program(x):
            return fw.nn.functional.gelu(x)
    elif name == "python_helper":

        def helper(x):
            return x.sin() * 0.5

        def program(x):
            return helper(x) + helper(x + 0.25)
    elif name == "custom_autograd":

        class Cubic(fw.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x * x * x

            @staticmethod
            def backward(ctx, gradient):
                (x,) = ctx.saved_tensors
                return gradient * 3 * x * x

        def program(x):
            return Cubic.apply(x)

        def inputs(v):
            return (matrix(v, grad=True),)
    elif name == "scalar_guard":

        def program(x, factor):
            return (x * factor).sin()

        def inputs(v):
            return matrix(v), 1.25 if v == 0 else -2.0
    elif name == "bfloat16_roundtrip":

        def program(x):
            reduced = x.to(dtype=fw.bfloat16) * 0.5
            return reduced, reduced.to(dtype=fw.float32)
    elif name == "device_roundtrip":

        def program(x):
            return (x.to(device="cpu") * 1.25).to(device=public_device).relu()
    else:
        raise ValueError(f"unknown corpus case: {name}")

    def fresh_inputs(variant, sample=0):
        nonlocal input_epoch
        input_epoch = sample
        try:
            return inputs(variant)
        finally:
            input_epoch = 0

    return Program(program, fresh_inputs, parameters)
