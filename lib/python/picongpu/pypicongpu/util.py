"""
This file is part of the PIConGPU.
Copyright 2021-2022 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from typeguard import typechecked
import typing
import logging

attr_cnt = 0


# note: type_ may be either a type, or a definition by typing
# depending on the python version the type of typing.XXXX is different
# (_GenericMeta vs. GenericMeta) -- so we compute it on the fly
@typechecked
def build_typesafe_property(
        type_: typing.Union[type, type(typing.List[int])],
        name: typing.Optional[str] = None) -> property:
    if name is None:
        global attr_cnt
        name = str(attr_cnt)
        attr_cnt += 1
    # don't use private prefix '__' to avoid name mangling
    actual_var_name = 'magic_string_private_____{}'.format(name)

    @typechecked
    def getter(self) -> type_:
        if not hasattr(self, actual_var_name):
            raise AttributeError('variable is not initialized')
        return getattr(self, actual_var_name)

    @typechecked
    def setter(self, value: type_):
        setattr(self, actual_var_name, value)

    return property(getter, setter)


@typechecked
def unsupported(name: str, value: typing.Any = 1,
                default: typing.Any = None) -> None:
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
