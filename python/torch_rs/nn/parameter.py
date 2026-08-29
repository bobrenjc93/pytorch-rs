class Parameter:
    r"""A Tensor marker type to identify module parameters.

    This lightweight compatibility class is sufficient for APIs that need to
    distinguish parameters via ``isinstance(value, torch.nn.Parameter)``.
    """

    def __init__(self, data=None, requires_grad=True):
        self.data = data
        self.requires_grad = requires_grad

