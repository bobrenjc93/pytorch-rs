from . import data as data


def set_module(obj, mod):
    """
    Set the module attribute on a python object for a given object for nicer printing
    """
    if not isinstance(mod, str):
        raise TypeError("The mod argument should be a string")
    obj.__module__ = mod
