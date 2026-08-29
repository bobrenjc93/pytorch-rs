class Parameter:
    def __init__(self, data=None, requires_grad=True):
        self.data = data
        self.requires_grad = requires_grad
