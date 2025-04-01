"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Julian Lenz
License: GPLv3+
"""

import typeguard
import typing
import logging

attr_cnt = 0


# note: type_ may be either a type, or a definition by typing
# depending on the python version the type of typing.XXXX is different
# (_GenericMeta vs. GenericMeta) -- so we compute it on the fly
@typeguard.typechecked
def build_typesafe_property(
    type_: typing.Union[type, type(typing.List[int])], name: typing.Optional[str] = None
) -> property:
    if name is None:
        global attr_cnt
        name = str(attr_cnt)
        attr_cnt += 1
    # don't use private prefix '__' to avoid name mangling
    actual_var_name = "magic_string_private_____{}".format(name)

    @typeguard.typechecked
    def getter(self) -> type_:
        if not hasattr(self, actual_var_name):
            raise AttributeError("variable is not initialized")
        return getattr(self, actual_var_name)

    @typeguard.typechecked
    def setter(self, value: type_):
        setattr(self, actual_var_name, value)

    return property(getter, setter)


@typeguard.typechecked
def unsupported(name: str, value: typing.Any = 1, default: typing.Any = None) -> None:
    """
    Print a msg that the feature/parameter/thing is unsupported.

    If 2nd param (value) and 3rd param (default) are set:
    supress msg if value == default

    If 2nd param (value) is set and 3rd is missing:
    supress msg if value is None

    If only 1st param (name) is set: always print msg

    :param name: name of the feature/parameter/thing that is unsupported
    :param value: If set: only print warning if this is not None
    :param default: If set: check value against this param instead of none
    """

    if value != default:
        logging.warning("unsupported: {}".format(name))


class SelfRegistering:
    # IMPORTANT: This is a mutable type ON PURPOSE!
    # We will let our children register themselves by mutating this instance.
    _names = []

    # We have this as a "backup" because subclasses will have a real name
    # but we still want to be able to check against the dummy name.
    _dummy_name = "base class -- has no name"

    # This is supposed to be set (and registered) by our children.
    _name = _dummy_name

    @classmethod
    def _register(cls):
        if cls._name not in cls._names:
            cls._names.append(cls._name)

    def __init_subclass__(cls):
        super().__init_subclass__()
        if SelfRegistering in cls.__bases__:
            cls._names = []
        if cls._name != cls._dummy_name:
            cls._register()
