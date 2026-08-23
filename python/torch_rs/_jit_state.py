# This private module outlives replacement imports of ``torch_rs.jit``.


onednn_fusion_enabled = globals().get("onednn_fusion_enabled", False)


def _incompatible_bool_error(value):
    try:
        value_repr = repr(value)
    except BaseException:
        value_repr = "<repr raised Error>"
    return TypeError(
        "_jit_set_llga_enabled(): incompatible function arguments. "
        "The following argument types are supported:\n"
        "    1. (arg0: bool) -> bool\n\n"
        f"Invoked with: {value_repr}"
    )


def _has_bool_slot(value):
    value_type = type(value)
    for base in type.__getattribute__(value_type, "__mro__"):
        namespace = type.__getattribute__(base, "__dict__")
        if "__bool__" in namespace:
            return True
    return False


def set_onednn_fusion_enabled(value):
    global onednn_fusion_enabled

    if value is None:
        converted = False
    elif _has_bool_slot(value):
        conversion_failed = False
        try:
            converted = bool(value)
        except BaseException:
            conversion_failed = True
        if conversion_failed:
            raise _incompatible_bool_error(value)
    else:
        raise _incompatible_bool_error(value)

    onednn_fusion_enabled = converted
