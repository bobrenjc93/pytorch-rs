"""TorchScript.

This module contains functionality to support the JIT's scripting frontend, notably:
    - torch.jit.script

This is not intended to be imported directly; please use the exposed
functionalities in `torch.jit`.
"""

import collections


Attribute = collections.namedtuple("Attribute", ["value", "type"])

Attribute.__doc__ = """
    This method is a pass-through function that returns `value`, mostly
    used to indicate to the TorchScript compiler that the left-hand side
    expression is a class instance attribute with type of `type`. Note that
    `torch.jit.Attribute` should only be used in `__init__` method of `jit.ScriptModule`
    subclasses.

    Though TorchScript can infer correct type for most Python expressions, there are some cases where
    type inference can be wrong, including:

    - Empty containers like `[]` and `{}`, which TorchScript assumes to be container of `Tensor`
    - Optional types like `Optional[T]` but assigned a valid value of type `T`, TorchScript would assume
      it is type `T` rather than `Optional[T]`

    In eager mode, it is simply a pass-through function that returns `value`
    without other implications.

    Example:

    .. testcode::

        import torch
        from typing import Dict

        class AttributeModule(torch.jit.ScriptModule):
            def __init__(self) -> None:
                super().__init__()
                self.foo = torch.jit.Attribute(0.1, float)

                # we should be able to use self.foo as a float here
                assert 0.0 < self.foo

                self.names_ages = torch.jit.Attribute({}, Dict[str, int])
                self.names_ages["someone"] = 20
                assert isinstance(self.names_ages["someone"], int)

        m = AttributeModule()
        # m will contain two attributes
        # 1. foo of type float
        # 2. names_ages of type Dict[str, int]

    .. testcleanup::

        del AttributeModule
        del m

    Note: it's now preferred to instead use type annotations instead of `torch.jit.Attribute`:

    .. testcode::

        import torch
        from typing import Dict

        class AttributeModule(torch.nn.Module):
            names: Dict[str, int]

            def __init__(self) -> None:
                super().__init__()
                self.names = {}

        m = AttributeModule()

    .. testcleanup::

        del AttributeModule
        del m

    Args:
        value: An initial value to be assigned to attribute.
        type: A Python type

    Returns:
        Returns `value`
"""
