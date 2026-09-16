# AST-extracted historical guard only; never imported as a module.
# Source: 6f0123a2cc521af9fab3dc19f1c256b28684f0db:python/torch_rs/_compile_pointwise.py
# Source blob SHA256: 68c8ed16cd1346ad9a3a908896b19a7b78eb33f7fcd68c826e49d84b1af12053
# Retained locally so controls also run in shallow/source-only checkouts.
_METHOD_GUARDS = tuple(((cls, name, cls.__dict__.get(name, _MISSING)) for cls in (_ROOT.Tensor, _ROOT.Tensor.__base__) for name in _METHODS))

def guard():
    for cls, name, expected in _METHOD_GUARDS:
        if cls.__dict__.get(name, _MISSING) is not expected:
            unsupported('patched Tensor operation binding: ' + name)
