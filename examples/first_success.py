import torch_rs as torch

x = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])
bias = torch.ones([2, 2])
result = torch.relu(x + bias)
delta = torch.sub(input=bias, other=x)
ratio = x.div(bias)

assert result.tolist() == [[0.0, 3.0], [4.0, 0.0]]
assert delta.tolist() == [[2.0, -1.0], [-2.0, 5.0]]
assert ratio.tolist() == [[-1.0, 2.0], [3.0, -4.0]]
